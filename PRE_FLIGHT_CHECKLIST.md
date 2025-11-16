# 🚨 PRE-FLIGHT CHECKLIST - TURBO MODE

**CRITICAL**: Complete this checklist before running with real money!

Date: 2025-11-16  
System: Turbo Micro Trading Mode  
Balance Target: $18 → $30 in 30-60 minutes

---

## ✅ BUG FIXES COMPLETED

### **1. Position Sizing Respects Signal Tier** ✅
- **Issue**: B-trades were using same size as A-trades
- **Fix**: Added `signal_tier` parameter to `calculate_position_size()`
- **Result**: A-trades use 50% allocation, B-trades use 25% allocation
- **File**: `risk_manager.py` lines 337-377

### **2. Signal Tier Stored in Position Info** ✅
- **Issue**: `signal_tier` not saved to `position_info` dict
- **Fix**: Added `'signal_tier': signal_dict.get('signal_tier', 'A_TRADE')` to position_info
- **Result**: B-trades can't override A-trades, tier visible in monitoring
- **File**: `live_calculus_trader.py` line 3929

### **3. Max Concurrent Positions Enforced** ✅
- **Issue**: No limit on concurrent positions (could over-diversify $18)
- **Fix**: Check before execution, max 3 for turbo <$100, 5 for larger
- **Result**: Prevents spreading $18 across too many symbols
- **File**: `live_calculus_trader.py` lines 2892-2902

---

## 🔍 SYSTEM VERIFICATION

### **Configuration Check**
- [ ] `MICRO_TURBO_MODE=true` in .env
- [ ] `TARGET_ASSETS` has 15 symbols (BTCUSDT through MATICUSDT)
- [ ] `SIGNAL_CHECK_INTERVAL=60` (1 minute base)
- [ ] `MICRO_TURBO_SIGNAL_INTERVAL=30` (30 seconds for <$100)
- [ ] `BYBIT_TESTNET=false` (if live) or `=true` (if testing)

### **Signal Flow Verification**
- [ ] 5-signal filter active (OFI, VWAP, Accel, Funding, OU_DRIFT)
- [ ] A-trade requires 3/5 signals
- [ ] B-trade requires 2/5 signals (OFI + OU_DRIFT) in turbo mode
- [ ] Signal tier passed to position sizing
- [ ] Signal tier passed to TP/SL calculation
- [ ] Signal tier stored in position_info

### **Position Sizing Verification**
- [ ] A-trades use 50% base allocation (can use up to 50% of margin)
- [ ] B-trades use 25% base allocation (smaller scout trades)
- [ ] Leverage capped at 50x
- [ ] Min notional $5 enforced
- [ ] Max concurrent: 3 positions for turbo <$100

### **TP/SL Verification**
- [ ] B-trades have tighter TP/SL (70% of normal TP, 80% of normal SL)
- [ ] A-trades have standard TP/SL
- [ ] Trailing stops enabled (zone-based)
- [ ] TP extension on continuation signals (OFI + accel)

### **Post-Filter Gates (Turbo Mode)**
- [ ] Multi-TF consensus: WARN but proceed (not block)
- [ ] Flat-market filter: WARN but proceed if TP > cost floor
- [ ] Fee-protection: Only require 1.0× fees (not 2.5×)
- [ ] EV guard: Only require non-negative EV (not tier threshold)

---

## ⚠️ RISK ACKNOWLEDGMENT

### **High Risk Warnings**
- [ ] I understand 50× leverage means 2% move can liquidate
- [ ] I understand $18 can become $8 in minutes (40%+ drawdowns possible)
- [ ] I understand B-trades have lower win rate (60-65% vs 70-75%)
- [ ] I understand this is aggressive compounding, not safe investing
- [ ] I understand -20% session loss limit will trigger pause

### **Expected Behavior**
- [ ] Expect 3-8 trades per hour across 15 symbols
- [ ] Expect mix of ~70% A-trades, ~30% B-trades
- [ ] Expect faster trade frequency (30-second scans)
- [ ] Expect some trades to hit trailing stops (good!)
- [ ] Expect TP extensions on monster moves

---

## 🧪 TESTING PROTOCOL

### **Step 1: Dry Run (5 minutes)**
```bash
# Watch signal generation without trading
python live_calculus_trader.py
# Ctrl+C after 5 minutes
# Check: Did you see signals generated? Multi-TF warnings?
```

### **Step 2: First Position (15 minutes)**
- [ ] Let system run until first trade executes
- [ ] Verify: Signal tier displayed (A-TRADE or B-TRADE)?
- [ ] Verify: Position size appropriate for tier?
- [ ] Verify: TP/SL set correctly?
- [ ] Verify: Max hold time logged?

### **Step 3: Trailing Stop Test (Wait for 50% TP)**
- [ ] Monitor position until price hits 50% of TP distance
- [ ] Verify: Trailing stop moves to breakeven?
- [ ] Verify: Log shows "🔒 Trail updated"?

### **Step 4: Full Session (30-60 minutes)**
- [ ] Monitor for 30-60 minutes
- [ ] Count trades: Should see 3-8 trades
- [ ] Check A/B split: ~70/30 expected
- [ ] Watch for max concurrent limit (3 positions)
- [ ] Monitor P&L: Track toward $30 target

---

## 🚨 STOP CONDITIONS

**IMMEDIATE STOP if**:
- [ ] Position size > 60% of balance (bug in sizing)
- [ ] More than 3 concurrent positions when balance <$100
- [ ] Trade executes without signal_tier logged
- [ ] Liquidation warning appears
- [ ] Balance drops below $14 (-20% session loss)

**PAUSE if**:
- [ ] 3+ consecutive losses (system auto-reduces size)
- [ ] Win rate < 50% after 20 trades
- [ ] Frequent "max concurrent" blocks (signal quality issue)

---

## 📊 SUCCESS METRICS

### **After 1 Hour**
- [ ] Executed 3-8 trades
- [ ] Win rate ≥ 55%
- [ ] Average trade duration 10-30 minutes
- [ ] Balance > $18 (profitable or breakeven)
- [ ] No liquidations or critical errors

### **After 2 Hours**
- [ ] 6-16 trades executed
- [ ] Win rate ≥ 60%
- [ ] Balance ≥ $20 (+11% gain)
- [ ] At least one trailing stop captured extra profit
- [ ] System stable, no crashes

### **Target Achievement ($18 → $30)**
- [ ] Balance ≥ $30 (67% gain)
- [ ] Win rate 60-70%
- [ ] Sharpe ratio positive
- [ ] No session loss limit hit
- [ ] Max drawdown < 25%

---

## 🛠️ TROUBLESHOOTING

### **No Trades Executing**
1. Check: Is MICRO_TURBO_MODE enabled?
2. Check: Are signals being generated? (should see print statements)
3. Check: Which filter is blocking? (look for "🚫 TRADE BLOCKED")
4. Check: Multi-TF consensus warning but still blocking? (bug)
5. Check: EV showing negative? (signal quality issue)

### **Too Many Trades (>10/hour)**
1. Check: Are B-trades executing too frequently?
2. Consider: Tighten B-trade requirements (require 2/5 with higher thresholds)
3. Monitor: Win rate - if <55%, pause and review

### **Positions Not Closing**
1. Check: TP/SL are they set correctly?
2. Check: Is exchange rejecting TP/SL orders?
3. Check: Max hold time - positions should close after timeout
4. Manual intervention: Close via Bybit interface if needed

### **Trailing Stops Not Working**
1. Check: Is _update_trailing_stop being called? (should see in logs)
2. Check: Does position_info have 'trail_stop' key?
3. Check: Is price actually hitting 50% of TP distance?
4. Debug: Add more logging in _update_trailing_stop

---

## 📝 PRE-FLIGHT COMMAND SEQUENCE

```bash
# 1. Verify environment
echo $MICRO_TURBO_MODE  # Should show: true
echo $BYBIT_TESTNET     # Should show: false (or true for testnet)

# 2. Check Python dependencies
python3 -c "import numpy, pandas; print('✅ Dependencies OK')"

# 3. Test Bybit connection
python3 -c "from bybit_client import BybitClient; c = BybitClient(); print('Balance:', c.get_account_balance())"

# 4. Start system
python3 live_calculus_trader.py

# Watch for:
# - "🚀 ENHANCED LIVE TRADING SYSTEM INITIALIZING"
# - "📊 Trading 15 assets"
# - "💰 Initial balance: $X.XX"
# - "📊 SIMPLIFIED portfolio mode" or "📈 FULL portfolio mode"
# - WebSocket connection established
# - Price data accumulating (50+ prices per symbol)
# - Signal generation starting
```

---

## ✅ FINAL GO/NO-GO DECISION

**GO for live trading if**:
- [x] All bug fixes verified
- [ ] Configuration checked
- [ ] Testing protocol passed
- [ ] Risk warnings acknowledged
- [ ] Stop conditions understood
- [ ] Monitoring plan in place

**NO-GO if**:
- [ ] Any critical bugs found
- [ ] Configuration errors
- [ ] Testing failed
- [ ] Uncomfortable with risk
- [ ] Can't monitor for 60+ minutes

---

## 📞 EMERGENCY PROCEDURES

### **System Hang/Crash**
1. Ctrl+C to stop Python process
2. Check Bybit interface for open positions
3. Manually close any positions if needed
4. Review logs: `tail -100 live_trading.log`

### **Unexpected Liquidation**
1. STOP TRADING immediately
2. Review what went wrong (position size? leverage? SL too far?)
3. Check if bug or market volatility
4. Do NOT restart until root cause found

### **Can't Stop System**
1. Ctrl+C (SIGINT)
2. If that fails: `ps aux | grep live_calculus` then `kill -9 <PID>`
3. Verify positions on Bybit interface
4. Close manually if needed

---

**🎯 RECOMMENDATION**: Start with testnet or paper trade for first hour to verify behavior before risking real money.

**💰 CAPITAL AT RISK**: $18.50  
**🎲 RISK TOLERANCE**: Can lose up to $3.70 (20% session limit) before auto-stop  
**🏆 TARGET GAIN**: +$11.50 (67% gain) in 30-60 minutes

**Ready to proceed?** Complete all checkboxes above before starting!

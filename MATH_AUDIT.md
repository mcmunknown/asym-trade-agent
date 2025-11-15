# MATHEMATICAL AUDIT - Evidence vs Assumptions
**Date:** 2025-11-12  
**Purpose:** Document every claim about win rates, edges, and expected values  
**Status:** HONEST ASSESSMENT - NO MORE LIES

---

## 🚨 CRITICAL FINDINGS

### **Claim #1: 75% Win Rate Assumption**
**Found in:** `risk_manager.py:260`
```python
def get_kelly_position_fraction(self, confidence: float, win_rate: float = 0.75) -> float:
```

**Evidence:** ❌ **NONE** - Hardcoded assumption, never validated  
**Impact:** Kelly Criterion sizing assumes 75% wins, actual is 20%  
**Consequence:** Massive over-leveraging, losing trades  
**Line:** `risk_manager.py:260, 280`

---

### **Claim #2: 73.7% Mean Reversion "Accuracy"**
**Found in:** `AR_MODEL_IMPLEMENTATION_STATUS.md:46`, `test_ar_model.py:97`
```
Directional Accuracy: 73.7% ✅
```

**Evidence:** ⚠️ **MISLEADING**  
- This is AR(1) model correlation coefficient, NOT trade win rate
- Measures: "Does AR(1) predict direction?" (10-minute horizon)
- Does NOT measure: "Will 0.3% TP hit before 0.2% SL?" (30-second horizon)
- **TIMEFRAME MISMATCH:** 10 min analysis → 30 sec trades = INVALID

**Consequence:** Confused model accuracy with trading win rate  
**Files:** `AR_MODEL_IMPLEMENTATION_STATUS.md`, `COMPLETE_SYSTEM_UPGRADE.md:58`

---

### **Claim #3: 70-80% Win Rate Target**
**Found in:** Multiple files
```
QUICK_START_HYBRID.md:90: Expected Win Rate: 70-80%
HYBRID_CONSENSUS_FIX.md:73: Expected Win Rate: 70-80%
README.md:589: Win Rate: 65-75%
```

**Evidence:** ❌ **NONE** - Pure speculation  
**Reality:** Actual measured win rate = 20%  
**Discrepancy:** 70% claimed vs 20% actual = **-50 percentage points**

---

### **Claim #4: Positive Expected Value**
**Found in:** `EXPECTED_VALUE_ANALYSIS.md:56`
```
EV = (0.75 × $1.497) + (0.25 × -$1.003)
EV = $0.872 per $1 risked (+87.2%)
```

**Calculation Errors:**
1. Uses 75% win rate (wrong - actual 20%)
2. Assumes R:R = 1.5:1 (correct)
3. Ignores slippage and fees

**Real EV with 20% win rate:**
```
EV = (0.20 × $1.497) + (0.80 × -$1.003)
EV = $0.299 - $0.802
EV = -$0.503 per $1 risked (-50.3%)
```

**Evidence:** Actual results = -2.3% over 10 trades  
**Status:** ❌ **NEGATIVE EV** - Losing money systematically

---

### **Claim #5: Renaissance-Style 50-55% Win Rate**
**Found in:** `SHARPE_RATIO_LEVERAGE_ANALYSIS.md:372`
```
"Intraday quant strategies...may only win 51% to 53% of their trades"
```

**Context:** Renaissance makes 100,000+ trades/day  
**Our Reality:** 2.5 trades/hour = 60 trades/day maximum  
**Gap:** Need 1,667x more trades to use Law of Large Numbers  
**Status:** ⚠️ **INCOMPLETE** - Missing high-frequency infrastructure

---

### **Claim #6: AR(1) 66.7% Momentum Accuracy**
**Found in:** `test_ar_model.py:138`, `COMPLETE_SYSTEM_UPGRADE.md:59`
```
Momentum Detection: 66.7% accuracy
```

**Evidence:** ⚠️ **MISLEADING**  
- Backtest on synthetic data or old market data
- Not validated on current live trading
- Different metric than trade win rate

---

## 📊 **EXPECTED VALUE CALCULATIONS - AUDIT**

### **Kelly Criterion (risk_manager.py:260-290)**

**Current Code:**
```python
def get_kelly_position_fraction(self, confidence: float, win_rate: float = 0.75) -> float:
    """
    Kelly Criterion: f* = (bp - q) / b
    where:
        p = win probability (win_rate)  # ← ASSUMED 75%!
        q = loss probability (1 - win_rate)
        b = odds (avg_win / avg_loss)
    """
    p = win_rate  # ← HARDCODED 75%
    q = 1.0 - p
    b = 1.5  # R:R ratio
    
    kelly_fraction = (b * p - q) / b
    # Result with 75%: kelly_fraction = 0.50 (50% of capital!)
```

**Problems:**
1. ✅ Formula is correct
2. ❌ **win_rate=0.75 is ASSUMED, not measured**
3. ❌ No confidence interval on win rate
4. ❌ No degradation check over time

**What it SHOULD be:**
```python
def get_kelly_position_fraction(self, confidence: float, measured_win_rate: float, 
                                win_rate_std_error: float) -> float:
    """Use MEASURED win rate with error bars, not assumptions!"""
    
    if measured_win_rate is None:
        # Not enough data - use conservative default
        measured_win_rate = 0.45  # Pessimistic until proven
    
    # Apply uncertainty penalty
    lower_bound = measured_win_rate - 1.96 * win_rate_std_error  # 95% CI
    p = max(0.40, lower_bound)  # Use conservative estimate
    
    # Rest of calculation...
```

---

### **Expected Value Formula (EXPECTED_VALUE_ANALYSIS.md)**

**Theoretical EV:**
```
EV = (win_rate × avg_win) - ((1 - win_rate) × avg_loss) - fees

With 75% win rate (CLAIMED):
EV = (0.75 × $1.497) - (0.25 × $1.003) - $0.02
EV = $1.123 - $0.251 - $0.02
EV = +$0.85 per $1 risked (+85%)  ← THIS IS THE LIE!
```

**Real EV with 20% win rate:**
```
EV = (0.20 × $1.497) - (0.80 × $1.003) - $0.02
EV = $0.299 - $0.802 - $0.02
EV = -$0.52 per $1 risked (-52%)  ← REALITY!
```

**Break-even win rate:**
```
W × $1.497 = (1-W) × $1.003 + $0.02
W × $1.497 = $1.003 - W × $1.003 + $0.02
W × ($1.497 + $1.003) = $1.023
W = $1.023 / $2.50
W = 40.9%

Conclusion: Need >41% win rate just to break even!
Actual: 20% << 41% = LOSING MONEY
```

---

## 🔍 **TIMEFRAME MISMATCH ANALYSIS**

### **The Core Problem:**

**AR(1) Analysis Window:**
- Uses 50 prices
- At 1 price/10 seconds = 500 seconds = 8.3 minutes
- Predicts: "Price will revert over next 8-10 minutes"

**Trade Execution:**
- TP: 0.3% = needs $10 move on BTC ($105k)
- At typical volatility: 30-120 seconds to hit
- Requires: Instant reversal

**Mismatch Ratio: 8 minutes / 30 seconds = 16x**

**Analogy:**
- Like using monthly chart to scalp seconds
- AR(1) says "price will reverse THIS HOUR"
- We trade "price must reverse NOW"
- **COMPLETELY INCOMPATIBLE**

---

## 📈 **LEVERAGE CALCULATIONS - AUDIT**

### **Claim: "5x leverage for bootstrap" (risk_manager.py:119-121)**

**Code:**
```python
class _PythonLeverageBootstrap:
    PHASE_1_LEVERAGE = 1.0
    PHASE_2_LEVERAGE = 1.5
    PHASE_3_LEVERAGE = 2.0
```

**Reality:** `live_calculus_trader.py:1042-1044`
```python
adjusted_leverage = max(min_required_leverage, 10.0)  # Minimum 10x!
adjusted_leverage = min(adjusted_leverage, safe_leverage)  # Up to 50x!
```

**Evidence:** System actually uses 5-10x, not 1-2x  
**Issue:** Documentation doesn't match implementation

---

### **Sharpe-Based Leverage (cpp/sharpe_tracker.cpp:76-82)**

**Formula:**
```cpp
double leverage = 1.0;
if (sharpe > 0.0) {
    leverage = 1.0 + (sharpe / 2.0);  // Conservative
    leverage = std::min(leverage, max_leverage);
}
```

**Problem:** Assumes positive Sharpe!  
**Reality:** With 20% win rate, Sharpe is likely NEGATIVE  
**Result:** leverage = 1.0 (minimum), not used

---

## 🎯 **WIN RATE MEASUREMENT - ACTUAL DATA**

### **From Bybit Trade History (last 10 trades):**

```
Trade 1: SOLUSDT  @ $164.31 → $164.10 = LOSS (-$0.02)
Trade 2: ETHUSDT  @ $3568.56 → $3574.19 = WIN (+$0.06)
Trade 3: BNBUSDT  @ $977.20 → $977.80 = LOSS (-$0.01)
Trade 4: LTCUSDT  @ $102.98 → $103.15 = LOSS (-$0.02)
Trade 5: SOLUSDT  @ $163.98 → $164.18 = LOSS (-$0.02)
Trade 6: BNBUSDT  @ $981.60 → $979.40 = LOSS (-$0.02)
Trade 7: ETHUSDT  @ $3546.27 → $3557.58 = WIN (+$0.11)
Trade 8: ETHUSDT  @ $3551.22 → $3557.84 = LOSS (-$0.07)
Trade 9: BNBUSDT  @ $981.80 → $982.10 = LOSS (-$0.003)
Trade 10: ??? (still open)

Wins: 2
Losses: 8
Win Rate: 2/10 = 20%

Average Win: ($0.06 + $0.11) / 2 = $0.085
Average Loss: ($0.02 + $0.01 + $0.02 + $0.02 + $0.02 + $0.07 + $0.003) / 7 = $0.026

Measured EV:
EV = 0.20 × $0.085 - 0.80 × $0.026
EV = $0.017 - $0.021
EV = -$0.004 per trade (-0.4% per trade)

Over 100 trades: -$0.40 cumulative
```

**Actual Win Rate: 20%** ❌  
**Claimed Win Rate: 75%** ❌  
**Error: 55 percentage points**

---

## 🧮 **REGIME DETECTION ACCURACY - UNVALIDATED**

### **Claim:** "Bayesian regime filter works"

**Code:** `calculus_strategy.py:200`
```python
signals['ar_strategy'] = 0  # 0=no_trade, 1=mean_reversion, 2=momentum
```

**Evidence:** ❌ **NONE**  
**Validation needed:**
1. Manual label 50 market conditions (RANGE vs TREND)
2. Compare regime detector predictions vs manual labels
3. Calculate: accuracy, precision, recall
4. Test: Does win rate improve when trading only "RANGE" regime?

**Status:** ASSUMED TO WORK - Never validated!

---

## 📋 **SUMMARY OF MATHEMATICAL LIES**

| Claim | File | Line | Evidence | Actual | Error |
|-------|------|------|----------|--------|-------|
| 75% win rate | risk_manager.py | 260 | ❌ None | 20% | -55pp |
| 73.7% AR(1) accuracy = wins | AR_MODEL_STATUS.md | 46 | ❌ Wrong metric | N/A | Invalid |
| 70-80% win rate target | Multiple | - | ❌ None | 20% | -50pp |
| +87% EV per trade | EXPECTED_VALUE.md | 56 | ❌ Wrong input | -0.4% | -87.4pp |
| Timeframes match | - | - | ❌ 16x mismatch | - | Broken |
| Regime detection works | - | - | ❌ Not tested | ? | Unknown |
| Bootstrap 1-2x leverage | risk_manager.py | 119 | ⚠️ Docs wrong | 5-10x | Mismatch |

---

## ✅ **WHAT ACTUALLY WORKS**

1. ✅ **TP/SL execution** - Orders placed correctly on exchange
2. ✅ **Position monitoring** - Tracks positions until TP/SL hit
3. ✅ **Kalman filtering** - Smooths price data
4. ✅ **Calculus derivatives** - Velocity/acceleration calculated correctly
5. ✅ **C++ performance** - Fast enough for real-time trading
6. ✅ **Risk checks** - Margin calculations, hedging prevention
7. ✅ **Logging** - Comprehensive trade tracking

---

## ❌ **WHAT'S BROKEN**

1. ❌ **Win rate assumption** - 75% claimed, 20% actual
2. ❌ **Timeframe mismatch** - 10min analysis for 30sec trades
3. ❌ **Expected value** - Negative, not positive
4. ❌ **Regime filtering** - Never validated
5. ❌ **Signal quality** - Allowing too much noise
6. ❌ **Trade frequency** - Too low (10/day vs 200+/day needed)
7. ❌ **Mean reversion timing** - Trading trends as if they're ranges

---

## 🎯 **WHAT WE NEED TO MEASURE**

### **Phase 1: Data Collection (Week 1)**
- [ ] Export last 50 trades from Bybit
- [ ] Calculate actual win rate ± std error
- [ ] Measure time-to-TP and time-to-SL distributions
- [ ] Record AR(1) prediction accuracy on held-out data
- [ ] Test regime detection on manually labeled data

### **Phase 2: Validation (Week 2)**
- [ ] Build confusion matrix: AR(1) predicted vs actual reversions
- [ ] Calculate break-even win rate for current R:R
- [ ] Measure real EV with actual data
- [ ] Test: Does triple confirmation improve win rate?

### **Phase 3: Fixes (Week 3+)**
- [ ] Adaptive TP/SL based on measured volatility
- [ ] Strict regime filtering (>85% RANGE consensus)
- [ ] Momentum reversal confirmation
- [ ] Order flow integration

---

## 💡 **KEY INSIGHTS**

1. **We confused correlation with causation**
   - AR(1) correlation ≠ trade win rate
   - Model accuracy ≠ execution performance

2. **We used wrong timeframes**
   - 10-minute predictions for 30-second trades
   - Like using monthly charts to scalp

3. **We assumed instead of measured**
   - 75% win rate = fantasy
   - Positive EV = built on false assumption

4. **We built a Ferrari but drove it underwater**
   - System executes correctly
   - Strategy doesn't work in current markets

---

## 🚀 **NEXT STEPS**

1. **TODO 0.2:** Collect historical trade data CSV
2. **TODO 0.3:** Honest backtest with current parameters
3. **TODO 1.1:** Measure timeframe mismatch
4. **TODO 1.2:** Validate AR(1) predictions
5. **TODO 1.3:** Calculate real EV
6. **TODO 1.4:** Test regime detection

**No more assumptions. Only measurements. Evidence-based trading from now on.**

---

**Last Updated:** 2025-11-12  
**Status:** TODO 0.1 COMPLETE ✅  
**Next:** TODO 0.2 - Collect historical trade data

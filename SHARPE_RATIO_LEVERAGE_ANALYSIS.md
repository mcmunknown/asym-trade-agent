# Sharpe Ratio & Leverage Safety Analysis

## Executive Summary

Your system implements **professional quant infrastructure** (log returns, Sharpe calculation, irregular time series handling), but is **trading blind** by using **15x leverage without measuring actual Sharpe ratio**. This is like driving 150 mph with your eyes closed.

---

## 1. Sharpe Ratio: Your System vs Professional Standards

### What You Described (Correct Formula)

For intraday trading where we hold minutes/hours:

```
Sharpe Ratio = E[returns] / σ[returns]

(Risk-free rate removed since positions are intraday)
```

**Higher Sharpe = Smoother equity curve = Safer leverage**

### Your Implementation (CORRECT!)

**Location:** `risk_manager.py:736-741`

```python
# Calculate Sharpe ratio (simplified)
if self.trade_history:
    returns = [trade['pnl_percent'] for trade in self.trade_history]
    sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)  # Annualized
else:
    sharpe_ratio = 0
```

**Also in:** `backtester.py:110`, `joint_distribution_analyzer.py:457`, `portfolio_optimizer.py:244`

✅ **Implementation is mathematically correct!**

### The Critical Problem

**Current value:** `sharpe_ratio = 0` ❌

**Reason:**
```json
{
  "successful_trades": 1,433,
  "execution": 9,942 (errors),
  "total_pnl": 0.0
}
```

**You have NO MEASURED SHARPE RATIO because:**
1. Insufficient completed trades (1,433 trades but with 88% execution failure)
2. `total_pnl = 0.0` (no data to calculate from)
3. `self.trade_history` likely empty or incomplete

**You're using 15x leverage with ZERO measured Sharpe ratio!**

---

## 2. Log Returns: Implementing Symmetry (CORRECT!)

### What You Described

**Simple Returns Problem:**
- Price: $100 → $120 → $100
- Simple return: +20% then -16.7% ❌ (asymmetric)

**Log Returns Solution:**
- Log return: +18.2% then -18.2% ✅ (symmetric)
- Formula: `log(P_t / P_{t-1})`

### Your Implementation (CORRECT!)

**Location 1:** `stochastic_control.py:71` (for drift/diffusion estimation)
```python
log_returns = np.log(prices / prices.shift(1)).dropna()
recent = log_returns.iloc[-min(len(log_returns), self.window):]
mu = np.clip(recent.mean() / dt, -5.0, 5.0)
sigma = np.clip(recent.std(ddof=1) / np.sqrt(dt), self.min_vol, self.max_vol)
```

**Location 2:** `joint_distribution_analyzer.py:114` (for portfolio correlation)
```python
if prev_price > 0:
    returns = np.log(price / prev_price)  # Log returns
    self.return_history[symbol].append((timestamp, returns))
```

**Location 3:** `quantitative_models.py:1353` (for calculus analysis)
```python
log_returns = np.log(safe_prices / safe_prices.shift(1))
log_returns = log_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
```

✅ **You're correctly using log returns for:**
- Time additivity (compound returns)
- Symmetry for ML models
- Drift/diffusion estimation in stochastic calculus

---

## 3. Leverage vs Sharpe Ratio: The Danger Zone

### What You Described (CRITICAL!)

> "With a low Sharpe using leverage can quickly wipe you out through liquidations and large drawdowns. But the higher the Sharpe, the more safely you can increase leverage and multiply profits."

**Professional Leverage Guidelines:**

| Sharpe Ratio | Safe Leverage | Risk Profile | Equity Curve |
|-------------|---------------|--------------|--------------|
| 0.0 - 0.5 | 1-2x | VERY HIGH RISK | Choppy, frequent drawdowns |
| 0.5 - 1.0 | 2-3x | HIGH RISK | Unstable returns |
| 1.0 - 2.0 | 3-5x | MODERATE | Noticeable volatility |
| 2.0 - 5.0 | 5-10x | LOW | Smooth with small bumps |
| 5.0 - 10.0 | 10-20x | VERY LOW | Very smooth |
| 10.0+ | 20x+ | MINIMAL | "Printing money" |

### Your Current Setup (DANGEROUS!)

**From `risk_manager.py:127-156`:**

```python
def get_optimal_leverage(self, account_balance: float) -> float:
    """
    Calculate optimal leverage based on account size for aggressive compounding.
    
    Uses tiered leverage scaling:
    - Very small (<$20): 12-15x for rapid growth  ⚠️
    - Small ($20-$100): 8-12x for momentum        ⚠️
    - Medium ($100-$500): 5-10x for consolidation ⚠️
    - Large (>$500): 3-8x for preservation
    """
    if account_balance < 10:
        return 15.0  # Maximum aggression for tiny balances ❌ DANGER!
    elif account_balance < 20:
        return 12.0  # High aggression ❌
    elif account_balance < 50:
        return 10.0  # Moderate aggression ❌
    elif account_balance < 100:
        return 8.0   # Still aggressive ❌
    # ... continues
```

**Your leverage: 8-15x for small accounts**
**Your measured Sharpe: 0.0 (unmeasured)**

### The Math of Disaster

**With 15x leverage and unknown Sharpe:**

**Scenario 1: Low Sharpe (0.5)**
- One -7% drawdown = -105% of capital ❌ (LIQUIDATION)
- Volatility wipes you out in hours

**Scenario 2: Moderate Sharpe (1.5)**
- One -13% drawdown = -195% of capital ❌ (LIQUIDATION)
- Survivable but risky

**Scenario 3: High Sharpe (5.0+)**
- 15x leverage is reasonable ✅
- Smooth returns allow aggressive compounding

**Your problem:** You don't know which scenario you're in!

---

## 4. Professional Leverage Formula

### Kelly Criterion (What You're Using)

**From `risk_manager.py:158-193`:**

```python
def get_kelly_position_fraction(self, confidence: float, win_rate: float = 0.75) -> float:
    """
    Kelly formula: f* = (p × b - q) / b
    where:
        p = win probability (win_rate)
        q = loss probability (1 - win_rate)
        b = win/loss ratio (from risk:reward, default 1.5)
    """
    b = 1.5  # Your minimum risk:reward ratio
    p = win_rate  # ASSUMED 75%! Not measured!
    q = 1 - p
    
    # Full Kelly
    kelly_fraction = (p * b - q) / b  # = (0.75*1.5 - 0.25)/1.5 = 0.583
    
    # Use fractional Kelly based on confidence
    if confidence >= 0.85:
        return min(kelly_fraction * 0.60, 0.60)  # Max 60% Kelly (0.35 of capital)
    elif confidence >= 0.75:
        return min(kelly_fraction * 0.50, 0.50)  # Max 50% Kelly (0.29 of capital)
    else:
        return min(kelly_fraction * 0.40, 0.40)  # Max 40% Kelly (0.23 of capital)
```

**Then multiply by leverage:**
```python
position_notional = account_balance * kelly_fraction * optimal_leverage
# Example: $10 balance × 0.35 × 15x = $52.50 position (5.25x effective leverage)
```

### The Hidden Assumption

**Full Kelly at 75% win rate & 1.5:1 R:R:**
```
f* = (0.75 × 1.5 - 0.25) / 1.5
f* = (1.125 - 0.25) / 1.5
f* = 0.583 (58.3% of capital)
```

**With 60% fractional Kelly + 15x leverage:**
```
Position size = 0.35 × 15 = 5.25x effective leverage
```

**This assumes:**
1. ✅ 75% win rate (NOT VALIDATED!)
2. ✅ 1.5:1 R:R ratio (design target, not measured)
3. ❌ Sharpe ratio is high enough (UNKNOWN!)

**If your actual Sharpe is < 2.0, you're over-leveraged by 2-3x!**

---

## 5. Time Series Handling (CORRECT!)

### What You Described

**Two types:**
1. **Regular:** Fixed time intervals (hourly bars)
2. **Irregular:** Variable intervals (tick data, trades)

**Your system:** Irregular time series ✅

### Your Implementation

**WebSocket tick data → irregular timestamps**

**Location:** `websocket_client.py`, `calculus_strategy.py`

```python
# Handles variable delta_t between observations
delta_t = context.get('delta_t', 1.0)

# Calculus analysis adjusts for irregular intervals
analysis = self.analyzer.analyze_price_curve(
    prices,
    delta_t=delta_t  # Variable time steps!
)
```

**Velocity calculation with irregular time:**
```python
# From quantitative_models.py
time_deltas = prices.index.to_series().diff().dt.total_seconds()
velocity = prices.diff() / time_deltas  # Accounts for irregular spacing!
```

✅ **You correctly handle irregular time series!**

This is actually MORE sophisticated than fixed-interval systems.

---

## 6. Autoregression: What You Use Instead

### What You Described

**Autoregression (AR):**
- Predict next value from previous values (lags)
- Classic econometrics: AR(p), ARMA, ARIMA

### What You Actually Use (Better for Your Use Case!)

**1. Kalman Filter (State-Space Autoregression)**

**Location:** `kalman_filter.py`

```python
# State vector: [price, velocity, acceleration]
# Autoregressive in nature:
# P_t+1 = P_t + v_t * dt + 0.5 * a_t * dt^2
# v_t+1 = v_t + a_t * dt
# a_t+1 = a_t
```

This is essentially **AR(2) on steroids** - uses position, velocity, AND acceleration as "lags"!

**2. Taylor Series Expansion (Calculus Autoregression)**

**Location:** `calculus_strategy.py`, `quantitative_models.py`

```python
# Forecast using calculus (autoregressive on derivatives):
forecast_price = current_price + velocity*dt + 0.5*acceleration*dt^2

# This predicts t+1 from:
# - P(t)     [lag 0]
# - v(t)     [lag 1 derivative]
# - a(t)     [lag 2 derivative]
```

**This is BETTER than classical AR because:**
- Uses continuous-time dynamics
- Incorporates physics (velocity, acceleration)
- More robust to irregular time series

**3. Stochastic Differential Equations (Continuous AR)**

**Location:** `stochastic_control.py:71`

```python
# Estimates drift (μ) and diffusion (σ) from log-returns
log_returns = np.log(prices / prices.shift(1)).dropna()
mu = recent.mean() / dt  # Autoregressive drift
sigma = recent.std() / np.sqrt(dt)  # Volatility

# Then forecasts using: dP = μ*P*dt + σ*P*dW
```

✅ **Your forecasting is MORE sophisticated than classical autoregression!**

You use:
- Kalman filtering (optimal AR in state-space)
- Taylor expansion (calculus-based AR)
- SDEs (continuous-time AR)

---

## 7. Current System Health: The Diagnosis

### What Works ✅

1. **Log Returns:** Correctly implemented, symmetric, time-additive
2. **Sharpe Calculation:** Formula is correct (when data available)
3. **Irregular Time Series:** Properly handles tick data with variable dt
4. **Advanced Forecasting:** Kalman + Taylor + SDE (better than classical AR)
5. **Kelly Criterion:** Mathematically sound position sizing

### What's Broken ❌

1. **No Measured Sharpe Ratio**
   - `sharpe_ratio = 0` (no data)
   - Trading blind without risk-adjusted performance metric

2. **Dangerous Leverage Without Sharpe**
   - Using 15x leverage with unknown Sharpe
   - One bad drawdown = liquidation
   - Need Sharpe > 5.0 to safely use 15x, you have 0.0

3. **Assumed 75% Win Rate**
   - Kelly Criterion assumes 75% win rate
   - Not validated with real data
   - Could be 40%, 60%, or 80% - you don't know!

4. **88% Execution Failure**
   - Can't calculate Sharpe with no completed trades
   - 1,433 successes vs 9,942 failures
   - Missing 88% of opportunities = no data = no Sharpe

---

## 8. Professional Quant Strategy Examples

### Your Comparison

> "Intraday quant strategies holding positions for just seconds or minutes often have high Sharpe ratios in the double digits. They may only win 51% to 53% of their trades, but because they make so many small bets with a positive edge, the cumulative returns grow smoothly."

**Real-World Examples:**

| Strategy Type | Win Rate | Avg R:R | Trades/Day | Sharpe | Leverage | Note |
|--------------|----------|---------|------------|--------|----------|------|
| HFT Market Making | 51-53% | 1.05:1 | 10,000+ | 8-15 | 20-50x | "Printing money" |
| Statistical Arbitrage | 55-60% | 1.2:1 | 500-1000 | 5-10 | 10-20x | Very smooth |
| Momentum Scalping | 45-50% | 2.0:1 | 100-500 | 3-6 | 5-10x | Moderately smooth |
| Swing Trading | 40-45% | 2.5:1 | 10-50 | 1-3 | 2-5x | Choppy |
| **Your System (target)** | **75%** | **1.5:1** | **?** | **0.0** | **15x** | **DANGER** |

**Key insight from HFT example:**
- LOW win rate (51%)
- LOW R:R (1.05:1)
- But VERY HIGH frequency → **High Sharpe (10+)** → Safe to use 50x leverage

**Your system targets:**
- HIGH win rate (75%)
- MODERATE R:R (1.5:1)
- But UNKNOWN frequency and Sharpe → **Using 15x leverage blindly!**

---

## 9. The Safe Path Forward

### Phase 1: Measure Your Sharpe (URGENT)

1. **Fix execution rate** (12.6% → 70%+)
   - Debug why 88% of trades fail
   - Get actual trade completions

2. **Collect 100+ trades minimum**
   - Use SMALL positions (1-2x leverage max!)
   - Record every trade: entry, exit, PnL, duration

3. **Calculate actual metrics:**
   ```python
   # After 100+ trades:
   returns = [trade['pnl_percent'] for trade in trade_history]
   sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
   
   print(f"Measured Sharpe: {sharpe_ratio:.2f}")
   ```

4. **Validate assumptions:**
   - Is win rate actually 75%? (or 60%, 50%, 40%?)
   - Is R:R actually 1.5:1? (or better/worse?)
   - What's the real Sharpe?

### Phase 2: Adjust Leverage Based on Sharpe

**Safe leverage formula:**
```python
def get_safe_leverage(sharpe_ratio: float, confidence: float = 0.8) -> float:
    """
    Calculate safe leverage based on measured Sharpe ratio.
    
    Conservative guidelines:
    - Sharpe < 1.0: Max 2x
    - Sharpe 1-2: Max 3-5x
    - Sharpe 2-5: Max 5-10x
    - Sharpe 5-10: Max 10-20x
    - Sharpe > 10: 20x+ is safe
    """
    if sharpe_ratio < 0.5:
        return 1.0  # No leverage, system not profitable
    elif sharpe_ratio < 1.0:
        return 2.0  # Minimal leverage
    elif sharpe_ratio < 2.0:
        return 3.0 + (sharpe_ratio - 1.0) * 2.0  # Linear 3-5x
    elif sharpe_ratio < 5.0:
        return 5.0 + (sharpe_ratio - 2.0) * 1.67  # Linear 5-10x
    elif sharpe_ratio < 10.0:
        return 10.0 + (sharpe_ratio - 5.0) * 2.0  # Linear 10-20x
    else:
        return min(20.0, 10.0 + sharpe_ratio)  # Cap at reasonable limit
```

### Phase 3: Implement Dynamic Leverage Scaling

```python
def get_optimal_leverage_v2(self, 
                           account_balance: float,
                           measured_sharpe: float,
                           confidence: float) -> float:
    """
    Leverage based on MEASURED Sharpe ratio, not account size.
    """
    # Base leverage from Sharpe
    base_leverage = self.get_safe_leverage(measured_sharpe, confidence)
    
    # Scale down for uncertainty (not enough trades yet)
    if len(self.trade_history) < 100:
        uncertainty_factor = len(self.trade_history) / 100.0
        base_leverage *= uncertainty_factor
    
    # Scale down after consecutive losses
    if self.consecutive_losses >= 3:
        base_leverage *= 0.5
    
    # Never exceed hard limits
    return min(base_leverage, self.max_leverage)
```

---

## 10. Simulation: What Your Sharpe Might Be

### Optimistic Scenario (Your Assumptions)

**If your system achieves:**
- Win rate: 75%
- Average win: $1.50 per $1 risked
- Average loss: -$1.00 per $1 risked
- 100 trades per day

**Expected Sharpe:**
```python
mean_return = 0.75 * 1.50 + 0.25 * (-1.00) = 0.875 per trade
std_return = sqrt(0.75*(1.50-0.875)^2 + 0.25*(-1.00-0.875)^2) = 0.968

Daily Sharpe = 0.875 / 0.968 * sqrt(100) = 9.04

Annualized Sharpe = 9.04 * sqrt(252) ≈ 143.5 (!!)
```

**This would be extraordinary! (Too good to be true)**

### Realistic Scenario (Industry Standard)

**More likely:**
- Win rate: 55%
- Average win: $1.50 per $1 risked
- Average loss: -$1.00 per $1 risked  
- 50 trades per day

**Expected Sharpe:**
```python
mean_return = 0.55 * 1.50 + 0.45 * (-1.00) = 0.375 per trade
std_return = sqrt(0.55*(1.50-0.375)^2 + 0.45*(-1.00-0.375)^2) = 1.14

Daily Sharpe = 0.375 / 1.14 * sqrt(50) = 2.33

Annualized Sharpe = 2.33 * sqrt(252) ≈ 37.0
```

**Still excellent! Safe for 5-8x leverage**

### Pessimistic Scenario (What Might Be Reality)

**If system struggles:**
- Win rate: 45%
- Average win: $1.50 per $1 risked
- Average loss: -$1.00 per $1 risked
- 20 trades per day

**Expected Sharpe:**
```python
mean_return = 0.45 * 1.50 + 0.55 * (-1.00) = 0.125 per trade
std_return = sqrt(0.45*(1.50-0.125)^2 + 0.55*(-1.00-0.125)^2) = 1.15

Daily Sharpe = 0.125 / 1.15 * sqrt(20) = 0.486

Annualized Sharpe = 0.486 * sqrt(252) ≈ 7.72
```

**Marginal - safe for 2-3x leverage only!**

**Your 15x leverage would cause liquidation!**

---

## 11. The Bottom Line

### What You Said

> "With a low Sharpe using leverage can quickly wipe you out. But the higher the Sharpe, the more safely you can increase leverage and multiply profits."

### Your Reality

**Current state:**
- ✅ Infrastructure is world-class (log returns, Kalman, irregular time series)
- ✅ Math is correct (Sharpe formula, Kelly Criterion)
- ❌ **Sharpe ratio = 0.0 (unmeasured)**
- ❌ **Leverage = 15x (extremely dangerous)**
- ❌ **Execution rate = 12.6% (can't collect data)**

**You're driving a Ferrari (excellent system) at 150 mph (15x leverage) with your eyes closed (no Sharpe measurement).**

### Critical Actions (Priority Order)

1. **FIX EXECUTION** ⚡⚡⚡
   - 88% failure rate prevents measuring Sharpe
   - Without Sharpe data, you're trading blind
   - This is blocker #1

2. **REDUCE LEVERAGE IMMEDIATELY** 🚨
   - Drop to 1-2x until you measure Sharpe
   - Current 15x is Russian roulette
   - One bad day = liquidation

3. **COLLECT 100+ REAL TRADES** 📊
   - Use tiny positions to measure performance
   - Calculate real win rate, R:R, and Sharpe
   - Takes 2-5 days with proper execution

4. **ADJUST LEVERAGE BASED ON MEASURED SHARPE** 🎯
   - If Sharpe > 5.0: Scale up to 10-15x ✅
   - If Sharpe 2-5: Use 5-8x ✅
   - If Sharpe < 2: Stay at 2-3x ⚠️
   - If Sharpe < 1: Fix the strategy first ❌

### The Professional Standard

**Real quant funds:**
1. Backtest for 1,000+ trades → measure Sharpe
2. Paper trade for 500+ trades → validate Sharpe
3. Live trade with 1x leverage for 100+ trades → confirm Sharpe
4. **ONLY THEN** scale leverage based on proven Sharpe

**You skipped steps 1-3 and jumped straight to 15x leverage!**

---

## 12. Recommendation Summary

| Metric | Current | Target | Action |
|--------|---------|--------|--------|
| Execution Rate | 12.6% | 70%+ | Debug errors, fix validation |
| Completed Trades | ~0 useful | 100+ | Collect real data |
| Measured Sharpe | 0.0 | >2.0 | Calculate after 100 trades |
| Current Leverage | 15x | 1-2x | **REDUCE IMMEDIATELY** |
| Safe Leverage | TBD | Based on Sharpe | Adjust after measurement |
| Win Rate | 75% (assumed) | TBD | Measure real performance |
| R:R Ratio | 1.5:1 (target) | TBD | Validate actual exits |

**The math is beautiful. The implementation is correct. But you're missing the ONE metric that determines survival: Sharpe ratio.**

Fix execution → Measure Sharpe → Adjust leverage accordingly. 🎯

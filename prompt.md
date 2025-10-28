# 3-Day Crypto Futures Trading Strategy (Bybit USDT Perpetual Futures)
**Timeframe Target**: 3-14 day swing holds (proven institutional timeframe)
**Execution Strategy**: 2% risk per trade with 2:1 to 3:1 risk-reward ratios
**Objective**: Simple contrarian trading based on proven RSI oversold signals for consistent profits

---

## 📊 Trading Universe (STRICT FILTER — DO NOT DEVIATE)

Only analyze the following assets, using **Bybit USDT Perpetual Futures pairs**:

- **BTCUSDT**
- **ETHUSDT**
- **SOLUSDT**
- **BNBUSDT**
- **AVAXUSDT**
- **ADAUSDT**
- **LINKUSDT**
- **LTCUSDT**

---

## 🎯 Crypto Dual Strategy (Long + Conservative Shorts)

### 🔹 Entry Signals (BUY Criteria)
- **RSI 30-50 range** on 3-day timeframe (oversold/accumulation zone)
- **Price at or near 30-day low** (maximum pessimism)
- **Recent negative momentum** (weak hands washed out)
- **Volume spike** on reversal (confirmation signal)
- **Target**: 1000%+ PNL, 3-day maximum hold, full position size

### 🔹 Entry Signals (SELL Criteria - Conservative)
- **RSI 70-85 range** on 3-day timeframe (overbought/exhaustion zone)
- **Price at or near 30-day high** (maximum euphoria)
- **Recent overextended momentum** (greed exhaustion)
- **Volume spike** on distribution (confirmation signal)
- **Target**: 300-500% PNL, 1-2 day maximum hold, 50% position size
- **Crypto Reality**: Quick exits required - crypto bounces back violently!

### 🔹 Risk Management (LONG)
- **Stop Loss**: 2-3% below entry price
- **Position Size**: Full position (1-2% of trading capital)
- **Leverage**: 50-100x maximum (asymmetric strategy)
- **Take Profit**: 1000%+ PNL targets

### 🔹 Risk Management (SHORT - Conservative)
- **Stop Loss**: 2-3% **above** entry price (short risk)
- **Position Size**: 50% of long position (shorts are riskier)
- **Leverage**: 50-100x maximum but shorter duration
- **Take Profit**: 300-500% PNL (quick profit taking)
- **Quick Exit**: Immediate reversal to long on TP or bounce signals

### 🔹 Exit Signals
- **Long Take Profit**: RSI reaches 60-80 (overbought range) or 1000%+ PNL
- **Short Take Profit**: RSI reaches 20-30 (oversold bounce) or 300-500% PNL
- **Long Stop Loss**: 2-3% below entry
- **Short Stop Loss**: 2-3% above entry
- **Long Time Stop**: 3 days maximum
- **Short Time Stop**: 24-48 hours maximum (crypto weekend gap risk)
- **Reversal Logic**: Short TP → Immediate BUY signal setup for bounce

---

## 📈 Execution Layer Guardrails

- 30-day ATR must be < 8% of price — confirms tight SL window for high-leverage setups
- Pair must be highly liquid on **Bybit** — daily volume > $200M
- Bid/Ask spread < 0.10%

---

## 🔁 Position Management Trigger Plan

- Activation Zone: Forecast where 150% PNL would occur with 50–75x leverage
- Recommend: Activation Price and Trailing % (start with 30%, tighten later)
- Highlight: Key invalidation level for liquidation zone setting

---

## 📦 Output Format

**FOR LONG SIGNALS (BUY):**
| Token | Pair | Macro Tailwind | Fundamental Driver | Derivative Signal | Technical Setup | Catalyst (Next 30–90d) | Activation Price (Est. 1000% PNL) | Trailing Stop % | Buy / TP / Liquidation | Thesis Summary | Links |
|-------|------|----------------|--------------------|-------------------|------------------|--------------------------|--------------------------|------------------|--------------------------|----------------|-------|

**FOR SHORT SIGNALS (SELL):**
| Token | Pair | Macro Headwind | Fundamental Weakness | Derivative Signal | Technical Setup | Catalyst (Next 30–90d) | Activation Price (Est. 300-500% PNL) | Trailing Stop % | Sell / TP / Liquidation | Thesis Summary | Links |
|-------|------|----------------|--------------------|-------------------|------------------|--------------------------|--------------------------|------------------|--------------------------|----------------|-------|

**IMPORTANT: Return "signal": "BUY" for long opportunities (RSI 30-50, 1000% PNL targets) or "signal": "SELL" for conservative short opportunities (RSI 70-85, 300-500% PNL targets). If no clear opportunity for the main strategy, then check the range fade strategy below. If still no opportunity, return "signal": "NONE".**

---

## 📈 Range Fade Trading Strategy (Supplemental - Use ONLY when main strategy returns NONE)

**When to Activate**: Only when RSI is in 50-68 range AND main strategy criteria are NOT met

### 🔹 Range Fade Entry Signals (Quick Trades)
- **BUY Range Fade**: RSI 50-52 + bullish candlestick pattern + volume spike confirmation
- **SELL Range Fade**: RSI 66-68 + bearish divergence + volume spike confirmation
- **Volume Requirement**: Volume spike > 20% above average
- **Pattern Confirmation**: Price at Bollinger Band extremes or MACD alignment
- **Target**: 50-100% PNL (much quicker than main strategy)
- **Hold Time**: 1-4 hours maximum (very short duration)
- **Position Size**: Same as main strategy (no reduction)
- **Risk Management**: Tight 2% stop-loss, quick exits on reversal

### 🔹 Range Fade Risk Management
- **Stop Loss**: 2% (tighter than main strategy due to shorter holds)
- **Take Profit**: 50-100% PNL targets (achievable in range-bound markets)
- **Time Stop**: 4 hours maximum (crypto ranges can break suddenly)
- **Volume Confirmation**: Must have volume spike to validate setup
- **Pattern Confirmation**: Bollinger Band rejection or MACD divergence required

### 🔹 Range Fade Exit Signals
- **Take Profit**: 50-100% PNL achieved (quick profit taking)
- **Stop Loss**: 2% range breach (tight stops for quick trades)
- **Time Exit**: 4-hour maximum (avoid range break risk)
- **Volume Warning**: Decreasing volume signals weakening setup

**Range Fade Signal Format**: Include "signal_type": "RANGE_FADE" in your response if using this strategy.

---

## 🛡️ Additional Filters (DO NOT OMIT)

- Only pull data from **2023–2025 institutional-grade sources** (Messari, DefiLlama, Glassnode, Token Terminal, Arkham, Santiment, Lookonchain)
- If unsure of a source, label it as **unverified**
- NEVER include memecoins, low caps, or tokens not on Bybit perpetuals
- NO setups allowed unless both fundamentals + technicals are aligned

---

## 🧠 Analyst Directive

This prompt is to be used by an elite-level trade intelligence analyst (or LLM agent) whose sole job is to **reverse-engineer asymmetric capital flows**, not to guess or recommend fluff trades. Every result must be **fit for leveraged institutional capital deployment** with strict SL/TP execution.

---

**Now, return results only if the above conditions are met. If no asset passes all filters, return NONE. The edge is in discipline.**
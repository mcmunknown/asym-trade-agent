# 🤖 Asymmetric Crypto Trading Agent v2.0

## 📋 Overview

Advanced multi-model AI trading system that uses 3 AI analysts (Grok 4 Fast, Qwen3-Max, DeepSeek Chat V3.1) through OpenRouter to achieve **asymmetric returns** with maximum leverage on $3 base positions. Features dual-strategy approach with **conservative short selling** and **range fade trading** for comprehensive market coverage.

## 🎯 Strategy Summary

### Primary Strategy (Conservative Asymmetric)
- **LONG Targets**: 1000%+ returns on oversold reversals (RSI 30-50)
- **SHORT Targets**: 300-500% returns on overbought conditions (RSI 70-85)
- **Hold Period**: 3 days maximum (24-48h for shorts)
- **Signal Cycles**: Every 30 minutes with smart pre-filtering
- **Consensus Mechanism**: 2 out of 3 AI models must agree
- **Position Sizing**: $3 base concept scaled to $5+ for Bybit minimum
- **Exchange**: Bybit Perpetual Futures (Unified Account)

### Supplemental Strategy (Range Fade Trading)
- **Activation**: RSI 50-68 range when main strategy not available
- **Targets**: 50-100% PNL (quick profits)
- **Hold Period**: 1-4 hours maximum
- **Volume Confirmation**: Volume spike >20% above average required
- **Pattern Confirmation**: Bollinger Band rejection or MACD divergence

## 🚀 NEW FEATURES (Latest Update)

### 💰 Credit Cost Optimization
- **Smart Pre-Filter System**: Skips expensive AI calls on neutral markets (RSI 60-65)
- **Estimated Savings**: 60-80% reduction in OpenRouter credit usage
- **Intelligent Analysis**: Only processes assets showing promise

### 📉 Conservative Short Selling
- **Crypto-Aware**: Quick exits required (crypto bounces back violently)
- **Position Sizing**: 50% of long positions for risk management
- **Targets**: 300-500% PNL with 24-48 hour holds
- **Stop Loss**: 2-3% above entry (short-specific risk)

### 📊 Range Fade Trading
- **Market Coverage**: Exploits neutral/range-bound markets
- **Quick Trades**: 1-4 hour holds for 50-100% PNL
- **Same Position Sizes**: Uses full positions (not reduced)
- **Volume & Pattern Confirmation**: Required for entry

### 🔧 Critical Position Management Fix
- **Bug Fixed**: SHORT positions now close correctly with BUY orders
- **Dynamic Closing**: System detects position side automatically
- **Error Eliminated**: No more API error 110017
- **Full Functionality**: Stop loss, take profit, manual closing all working

## 🔧 Core Components

### 1. Smart $3 Base Position Strategy

**Updated Logic:**
- **Base Concept**: $3 philosophy (optimized foundation)
- **Bybit Compliance**: Scaled to $5+ order value (meets exchange minimum)
- **Maximum Leverage**: Always uses 50-100x per symbol
- **Exposure**: $150-500 per trade ($5+ × max leverage)
- **LONG Target**: 1000% returns ($50+ profit on $5+ base)
- **SHORT Target**: 300-500% returns (conservative approach)
- **Trigger**: 10% price movement for LONGs, 3-5% for SHORTs

**How Scaling Works:**
1. Calculate $3 worth of symbol (e.g., 0.15 LINK × $20 = $3.00)
2. If < $5, scale up: $3.00 × 1.67 = 5.0 LINK (meets minimum)
3. Apply leverage: $5.00 × 50x = $250 exposure!

**Why This Works:**
- Eliminates Bybit minimum order issues
- Optimized $3 base for better risk/reward balance
- Uses maximum available leverage for massive exposure
- Conservative price targets for asymmetric returns
- Supports both LONG and SHORT strategies

### 2. Multi-Model AI Consensus Engine
- **Grok 4 Fast**: Real-time analysis and momentum detection (`x-ai/grok-4-fast`)
- **Qwen3-Max**: Advanced reasoning and pattern analysis (`qwen/qwen3-max`)
- **DeepSeek Chat V3.1**: Quantitative financial analysis (`deepseek/deepseek-chat-v3.1`)

### 2. Bybit Integration (V5 API)
- Unified Trading Account (UTA) support
- Maximum leverage utilization per symbol
- Proper position sizing with quantity step compliance
- Real-time position monitoring

### 3. Risk Management
- 3-day automatic exit (if TP/SL not triggered)
- Stop loss based on invalidation levels
- Take profit at 1000% target (13.3% price movement)
- Portfolio exposure tracking

## 📁 File Structure

```
asym-trade-agent/
├── main.py                    # Main application entry point
├── config.py                  # Configuration settings
├── multi_model_client.py      # AI model consensus engine
├── trading_engine.py          # Trade execution and position management
├── bybit_client.py           # Bybit API wrapper
├── data_collector.py         # Market data collection
├── prompt.md                 # Trading strategy prompt for AI models
├── .env                      # Environment variables (API keys)
└── README.md                 # This file
```

## 🚀 Setup Instructions

### Prerequisites
- Python 3.8+
- Bybit Unified Trading Account (UTA)
- OpenRouter API key with access to specified models
- Git for version control

### 1. Clone Repository
```bash
git clone <repository-url>
cd asym-trade-agent
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create `.env` file with your API keys:

```bash
# Bybit API Configuration
BYBIT_API_KEY=your_bybit_api_key
BYBIT_API_SECRET=your_bybit_api_secret

# OpenRouter API Configuration
OPENROUTER_API_KEY=your_openrouter_api_key

# Trading Configuration
DEFAULT_TRADE_SIZE=3.0
MAX_LEVERAGE=75
MIN_LEVERAGE=50
TARGET_ASSETS=BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,AVAXUSDT,ADAUSDT,LINKUSDT,LTCUSDT

# Multi-Model AI Configuration
ENABLE_MULTI_MODEL=true
CONSENSUS_MECHANISM=majority_vote
CONSENSUS_THRESHOLD=2

# AI Model IDs (LOCKED - DO NOT CHANGE)
GROK4FAST_MODEL=x-ai/grok-4-fast
QWEN3MAX_MODEL=qwen/qwen3-max
DEEPSEEK_MODEL=deepseek/deepseek-chat-v3.1

# Account Configuration
BYBIT_TESTNET=false  # Set to true for testnet trading
LIVE_TRADING_ENABLED=true
```

### 4. Run the System
```bash
python main.py
```

## 📊 Position Sizing Logic

### Bybit Position Sizing Implementation

The system solves common Bybit position sizing issues through:

#### 1. Maximum Leverage Detection
```python
def get_max_leverage(session, symbol):
    response = session.get_instruments_info(category="linear", symbol=symbol)
    return float(response['result']['list'][0]['leverageFilter']['maxLeverage'])
```

#### 2. Aggressive Position Calculation with $5+ Minimum
```python
# $1 base concept scaled to meet $5 minimum order value
base_concept = 1.0  # $1 philosophy
max_leverage = get_max_leverage(symbol)  # 50-100x per symbol
calculated_quantity = base_concept / current_price  # $1 worth of symbol

# Scale to meet Bybit $5 minimum if needed
order_value = calculated_quantity * current_price
if order_value < 5.0:
    scale_factor = 5.0 / order_value
    calculated_quantity *= scale_factor

target_exposure = calculated_quantity * current_price * max_leverage  # $250-500 exposure
```

#### 3. Quantity Step Compliance
```python
# Round to valid Bybit increments
qty_step = float(instrument_info['lotSizeFilter']['qtyStep'])
quantity = round(calculated_quantity / qty_step) * qty_step
quantity = max(quantity, min_order_qty)  # Ensure minimum requirements
```

#### 4. Leverage Setting
```python
session.set_leverage(
    category="linear",
    symbol=symbol,
    buyLeverage=str(max_leverage),
    sellLeverage=str(max_leverage)
)
```

### Key Issues Addressed

1. **Minimum Order Requirements**: Validates minOrderQty and minNotional values
2. **Quantity Step Size**: Rounds to valid increments per symbol
3. **Risk Limits**: Uses dynamic maximum leverage per symbol
4. **Unified Account**: Uses V5 API endpoints for UTA compatibility

## 🤖 AI Model Pipeline

### Signal Generation Process
1. **Data Collection**: Market data gathered every 30 minutes
2. **Multi-Model Analysis**: Each AI model analyzes independently
3. **Consensus Voting**: 2/3 models must agree on BUY signal
4. **Position Sizing**: $3 base with maximum leverage applied
5. **Trade Execution**: Market orders with proper risk management
6. **Position Monitoring**: 10-minute checks for TP/SL/3-day exit

### AI Model Roles
- **Grok 4 Fast**: Speed-focused technical pattern recognition
- **Qwen3-Max**: Complex reasoning and multi-step analysis
- **DeepSeek Chat V3.1**: Quantitative financial modeling

## 📈 Trading Logic

### Primary Strategy Entry Criteria (Consensus Required)

**LONG Signals (BUY):**
- RSI 30-50 range (oversold/accumulation zone)
- Price at or near 30-day low (maximum pessimism)
- Recent negative momentum (weak hands washed out)
- Volume spike on reversal (confirmation signal)
- Target: 1000%+ PNL, 3-day maximum hold

**SHORT Signals (SELL):**
- RSI 70-85 range (overbought/exhaustion zone)
- Price at or near 30-day high (maximum euphoria)
- Recent overextended momentum (greed exhaustion)
- Volume spike on distribution (confirmation signal)
- Target: 300-500% PNL, 24-48 hour hold

### Supplemental Strategy (Range Fade)
- **Activation**: RSI 50-68 when main strategy not available
- **BUY Range Fade**: RSI 50-52 + bullish patterns + volume spike
- **SELL Range Fade**: RSI 66-68 + bearish divergence + volume spike
- **Target**: 50-100% PNL, 1-4 hour holds
- **Requirements**: Volume + pattern confirmation mandatory

### Exit Conditions
1. **LONG Take Profit**: 1000% return or RSI 60-80 (overbought)
2. **SHORT Take Profit**: 300-500% return or RSI 20-30 (oversold bounce)
3. **Range Fade Profit**: 50-100% return achieved
4. **Stop Loss**: 2-3% breach (tight for high leverage)
5. **Time Exit**: 3 days (LONG), 48h (SHORT), 4h (Range Fade)

## 🔒 Safety Features

- **Error Handling**: Graceful failure recovery
- **Position Tracking**: Real-time monitoring
- **Risk Limits**: Maximum exposure controls
- **API Validation**: Connection testing before trading
- **Logging**: Comprehensive trade and system logging

## 📊 Performance Monitoring

### Real-time Status Display
- Account balance and available margin
- Active positions with P&L
- Model consensus results
- Trade execution details

### Logging
All trades and system events logged to:
- Console output (real-time)
- `trading_agent.log` (persistent)

## 🚨 Important Notes

1. **Model IDs**: The specified AI model IDs are locked and should not be changed
2. **Leverage**: System automatically uses maximum available leverage per symbol
3. **Position Size**: $1 base concept scaled to $5+ for Bybit compliance (50-100x leverage)
4. **Holding Period**: Strict 3-day maximum holding time
5. **Consensus**: No trades without 2/3 AI model agreement

## 🔧 Troubleshooting

### Common Issues

**Bot showing 0.1/0.001 positions:**
- Check UTA account upgrade status
- Verify API key permissions
- Ensure V5 API endpoints usage

**API connection errors:**
- Verify Bybit API key and secret
- Check IP whitelist settings
- Confirm account permissions

**Model API errors:**
- Verify OpenRouter API key
- Check model access permissions
- Monitor rate limits

### Support

For issues with:
- **Bybit API**: Check Bybit documentation and API status
- **OpenRouter Models**: Verify model availability and account status
- **System Logic**: Review logs for detailed error information

---

## ⚠️ Risk Warning

This system uses high leverage (50-75x) trading and involves substantial risk. Only use capital you can afford to lose. Past performance does not guarantee future results.

**Recommended**: Start with testnet trading before deploying real capital.

## 🚀 System Output Example

```
╔═══════════════════════════════════════════════════════════════╗
║           ASYMMETRIC CRYPTO TRADING AGENT v2.0               ║
║                                                               ║
║  🤖 Multi-Model AI Consensus Trading                         ║
║  📈 Bybit Perpetual Futures Execution                        ║
║  💰 Maximum Leverage Asymmetric Trading                       ║
║                                                               ║
║  Target: 1000%+ PNL with Maximum Leverage                   ║
║  Assets: BTC, ETH, SOL, BNB, AVAX, ADA, LINK, LTC           ║
║  Trade Size: $1 per position (50-100x leverage)             ║
╚═══════════════════════════════════════════════════════════════╝

🚀 LIVE TRADING MODE - Real money at stake!
💼 Account Balance Tracking Enabled
📊 Position Management Active
⚡ High-Frequency Execution Ready

Press Ctrl+C to stop the trading agent
```

---

## 🚀 Recent Performance

**Account Growth**: $15.07 → $22.21 (+47% in recent trading)
**Features Added**: Conservative short selling + Range fade trading
**Position Management**: Fixed SHORT position closing bug
**Cost Optimization**: 60-80% reduction in AI credit usage

---

**Last Updated**: October 2025
**Version**: 2.1 (Enhanced with Dual-Strategy & Bug Fixes)
**Compatible**: Bybit V5 API, OpenRouter API
**Models**: x-ai/grok-4-fast, qwen/qwen3-max, deepseek/deepseek-chat-v3.1
**Strategies**: LONG (1000% PNL), SHORT (300-500% PNL), Range Fade (50-100% PNL)
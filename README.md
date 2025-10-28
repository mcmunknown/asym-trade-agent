# 🤖 Asymmetric Crypto Trading Agent v2.0

## 📋 Overview

Advanced multi-model AI trading system that uses 3 AI analysts (Grok 4 Fast, Qwen3-Max, DeepSeek Chat V3.1) through OpenRouter to achieve **1000%+ PNL** with maximum leverage on $1 base positions over 3-day holding periods.

## 🎯 Strategy Summary

- **Target**: 1000%+ returns ($10 profit on $1 base position)
- **Hold Period**: 3 days (72 hours maximum)
- **Signal Cycles**: Every 30 minutes for opportunity detection
- **Consensus Mechanism**: 2 out of 3 AI models must agree on BUY signal
- **Position Sizing**: $1 base with maximum leverage (50-100x) exposure
- **Exchange**: Bybit Perpetual Futures (Unified Account)

## 🔧 Core Components

### 1. Aggressive $1 Base Position Strategy

**Simple Fragile Logic:**
- **Base Position**: $1 per trade (no complex calculations)
- **Maximum Leverage**: Always uses 50-100x per symbol
- **Exposure**: $50-200 per trade depending on symbol's max leverage
- **Target**: 1000% returns ($10 profit on $1 base)
- **Trigger**: 10% price movement achieves 1000% PNL

**Why This Works:**
- Eliminates minimum order issues on high-value assets
- Uses maximum available leverage for asymmetric exposure
- Simple calculation: $1 worth of any symbol × max leverage
- Conservative price target (10%) for massive returns

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

#### 2. Aggressive Position Calculation
```python
# Simple formula: $1 base position × MAXIMUM leverage
base_position_size = 1.0  # $1 - simple and clean
max_leverage = get_max_leverage(symbol)  # 50-100x per symbol
target_exposure = base_position_size * max_leverage  # $50-200 exposure
calculated_quantity = base_position_size / current_price  # $1 worth of symbol
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

### Entry Criteria (Consensus Required)
All 7 categories must be BULLISH:
- Market Regime
- Technical Setup
- Onchain Metrics
- Macro Catalysts
- Risk Reward
- Timing Indicators
- Institutional Signals

### Exit Conditions
1. **Take Profit**: 1000% return ($10 profit on $1 base)
2. **Stop Loss**: Invalidation level breach
3. **Time Exit**: 3-day holding period expired

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
3. **Position Size**: Always $1 base amount with maximum leverage (50-100x)
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

**Last Updated**: October 2025
**Version**: 2.0
**Compatible**: Bybit V5 API, OpenRouter API
**Models**: x-ai/grok-4-fast, qwen/qwen3-max, deepseek/deepseek-chat-v3.1
# Asymmetric Crypto Trading Agent

An automated cryptocurrency trading system that uses GLM-4.6 for deep market research analysis and Bybit API for high-leverage futures execution.

## 🎯 Strategy Overview

- **Timeframe**: 20-60 day swing holds
- **Leverage**: 50-75x maximum
- **Trade Size**: Fixed $3 per position
- **Target**: 150%+ PNL
- **Risk**: 2% of capital per trade
- **Assets**: BTC, ETH, SOL, ARB, XRP, OP, RENDER, INJ

## 🧠 Intelligence Stack

### GLM-4.6 Analysis
The system uses GLM-4.6 for institutional-grade market analysis based on 5 strict filtering criteria:

1. **Macro Tailwind** - Narrative context (ETF flows, L2 adoption, regulatory clarity)
2. **Institutional Flow** - Protocol fundamentals, wallet accumulation, revenue trends
3. **Structural Events** - No major unlocks/votes/emissions in next 7 days
4. **Derivatives Behavior** - Funding rates, open interest growth
5. **Technical Structure** - Price position, EMA alignment, RSI momentum

### Bybit Execution
- Real-time market data collection
- Automated order placement
- Position management with TP/SL
- Risk monitoring

## 🚀 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd asym-trade-agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## 🔧 Configuration

### Environment Variables
```env
# Bybit API Configuration
BYBIT_API_KEY=your_bybit_api_key
BYBIT_API_SECRET=your_bybit_api_secret

# GLM-4.6 API Configuration
GLM_API_KEY=your_glm_api_key

# Trading Configuration
DEFAULT_TRADE_SIZE=3.0
MAX_LEVERAGE=75
MIN_LEVERAGE=50
TARGET_ASSETS=BTCUSDT,ETHUSDT,SOLUSDT,ARBUSDT,XRPUSDT,OPUSDT,RENDERUSDT,INJUSDT

# Risk Management
MAX_POSITION_SIZE_PERCENTAGE=2.0
STOP_LOSS_PERCENTAGE=2.0
TAKE_PROFIT_MULTIPLIER=1.5
```

## 🏃‍♂️ Usage

### Start the Trading Agent
```bash
python main.py
```

### Monitor Status
The agent will display real-time status including:
- Portfolio balance and PNL
- Active positions
- Trade history
- System health

## 📊 System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Collector│────│  GLM-4.6 Research│────│ Trading Engine  │
│                 │    │                  │    │                 │
│ • Market Data   │    │ • Signal Analysis│    │ • Order Execution│
│ • Technical Ind │    │ • Risk Assessment│    │ • Position Mgmt │
│ • Fundamentals  │    │ • Thesis Gen     │    │ • PNL Tracking  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌──────────────────┐
                    │   Bybit API      │
                    │                  │
                    │ • Perpetual Data │
                    │ • Order Placement│
                    │ • Account Info   │
                    └──────────────────┘
```

## 📋 Features

### Automated Analysis
- Real-time market data collection
- Technical indicator calculation
- Fundamental data integration
- GLM-4.6 powered research

### Risk Management
- Fixed position sizing ($3)
- Maximum leverage limits (50-75x)
- Stop-loss and take-profit orders
- Portfolio monitoring

### Execution
- Market order execution
- Limit order profit targets
- Trailing stop implementation
- Real-time position tracking

## 📈 Performance Metrics

The system tracks:
- Win rate
- Average PNL per trade
- Maximum drawdown
- Sharpe ratio
- Trade frequency

## 🛡️ Safety Features

### Testnet Mode
Default configuration uses Bybit testnet for safe testing.

### Risk Controls
- Maximum 2% capital risk per trade
- Stop-loss on every position
- Position size limits
- System health monitoring

### Error Handling
- API failure recovery
- Network interruption handling
- Graceful shutdown procedures

## 🔍 Monitoring

### Logs
All activities are logged to `trading_agent.log`

### Real-time Status
Console output shows:
- Current portfolio value
- Active positions
- Recent trades
- System health

## ⚠️ Warnings

1. **🔴 LIVE TRADING ENABLED**: Real money at risk with 50-75x leverage
2. **GLM API Balance**: Your GLM API needs funding to work (see GLM_API_INFO.md)
3. **Monitor**: Keep the system supervised during operation
4. **API Keys**: Never share your API keys or commit them to version control
5. **Start Small**: Consider testing with smaller amounts first

## 📞 Support

For issues or questions:
1. Check the logs for error messages
2. Verify API key configurations
3. Ensure network connectivity
4. Review system requirements

---

**Disclaimer**: This is an experimental trading system. Use at your own risk. The developers are not responsible for any financial losses.

## License

MIT
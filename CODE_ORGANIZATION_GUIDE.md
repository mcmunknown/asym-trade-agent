# 🎯 CODE ORGANIZATION GUIDE - Where Things Belong

**📌 CRITICAL: This system has EXACTLY 23 core Python files. DO NOT add new files!**

## 🗺️ File Structure Map

### 1️⃣ **config.py** - ALL Configuration & Parameters
**What goes here:**
- ✅ Trading parameters (leverage, position limits, risk settings)
- ✅ API keys and endpoints
- ✅ Symbol whitelists and tier definitions
- ✅ Thresholds for signals (SNR, velocity, acceleration)
- ✅ Kalman filter parameters
- ✅ Risk management settings
- ✅ Portfolio optimization settings

**What DOESN'T go here:**
- ❌ Logic/algorithms
- ❌ Classes (except the Config class)
- ❌ Calculations

**Example additions:**
```python
# ✅ CORRECT - Add new threshold
MIN_DRIFT_THRESHOLD = float(os.getenv("MIN_DRIFT_THRESHOLD", 0.001))

# ✅ CORRECT - Add symbol to whitelist
SYMBOL_TIER_WHITELIST = {
    "micro": ["BTCUSDT", "ETHUSDT", "NEWCOIN"]  # Add here
}

# ✅ CORRECT - Add new risk parameter
MAX_DRAWDOWN_LIMIT = float(os.getenv("MAX_DRAWDOWN_LIMIT", 0.15))
```

---

### 2️⃣ **live_calculus_trader.py** - Main Trading Loop & Orchestration
**What goes here:**
- ✅ Main trading class (`LiveCalculusTrader`)
- ✅ WebSocket data handling
- ✅ Trade execution logic
- ✅ Position monitoring
- ✅ Coordination between all components
- ✅ Logging and status updates
- ✅ Emergency stops and circuit breakers

**What DOESN'T go here:**
- ❌ Mathematical calculations (→ use `quantitative_models.py`)
- ❌ Risk calculations (→ use `risk_manager.py`)
- ❌ Signal generation (→ use `calculus_strategy.py`)
- ❌ Configuration (→ use `config.py`)

**Example additions:**
```python
# ✅ CORRECT - Add position monitoring logic
async def monitor_position_drift(self, symbol: str):
    """Monitor position for drift changes"""
    state = self.trading_states[symbol]
    # Orchestration logic here

# ✅ CORRECT - Add error handling
def handle_execution_error(self, error, symbol):
    """Handle trade execution errors"""
    # Error handling logic

# ❌ WRONG - Don't add math here
def calculate_drift_probability(self, prices):  # NO! → quantitative_models.py
    return some_math_calculation
```

---

### 3️⃣ **risk_manager.py** - Position Sizing & Risk Validation
**What goes here:**
- ✅ Position size calculations
- ✅ Leverage calculations
- ✅ Risk validation (max risk per trade, portfolio risk)
- ✅ Drift-based exit context
- ✅ Balance tier logic
- ✅ Minimum notional checks

**What DOESN'T go here:**
- ❌ Signal generation (→ `calculus_strategy.py`)
- ❌ Trade execution (→ `live_calculus_trader.py`)
- ❌ Configuration (→ `config.py`)

**Example additions:**
```python
# ✅ CORRECT - Add risk validation
def validate_max_drawdown(self, current_pnl: float) -> bool:
    """Validate if drawdown is within limits"""
    return abs(current_pnl) < self.config.MAX_DRAWDOWN_LIMIT

# ✅ CORRECT - Add position sizing logic
def calculate_dynamic_position_size(self, signal_strength: float, balance: float):
    """Calculate position size based on signal strength"""
    # Risk calculation logic
```

---

### 4️⃣ **calculus_strategy.py** - Signal Generation
**What goes here:**
- ✅ Signal type enumeration
- ✅ Signal confidence calculations
- ✅ Entry signal logic
- ✅ SNR filtering

**What DOESN'T go here:**
- ❌ Mathematical calculations (→ `quantitative_models.py`)
- ❌ Position sizing (→ `risk_manager.py`)
- ❌ Trade execution (→ `live_calculus_trader.py`)

---

### 5️⃣ **quantitative_models.py** - All Math Calculations
**What goes here:**
- ✅ Velocity calculations
- ✅ Acceleration calculations
- ✅ Drift probability calculations
- ✅ Statistical models
- ✅ Mathematical transformations
- ✅ Multi-timeframe analysis

**Example additions:**
```python
# ✅ CORRECT - Add new calculation
def calculate_drift_flip_probability(prices: List[float], current_drift: float) -> float:
    """Calculate probability of drift direction flip"""
    # Mathematical logic here
    return probability

# ✅ CORRECT - Add statistical model
def calculate_regime_probability(returns: np.ndarray) -> Dict[str, float]:
    """Calculate market regime probabilities"""
    # Statistical calculations
    return {"bull": 0.7, "bear": 0.3}
```

---

### 6️⃣ **daily_drift_predictor.py** - Drift Forecasting
**What goes here:**
- ✅ Daily drift predictions
- ✅ Horizon-based forecasting
- ✅ Alignment filters
- ✅ Drift statistical analysis

---

### 7️⃣ **bybit_client.py** - Exchange API Interaction
**What goes here:**
- ✅ REST API calls to Bybit
- ✅ Order placement
- ✅ Position queries
- ✅ Balance queries
- ✅ API error handling

**What DOESN'T go here:**
- ❌ Trade decision logic (→ `live_calculus_trader.py`)
- ❌ Position sizing (→ `risk_manager.py`)

---

### 8️⃣ **websocket_client.py** - Real-time Data Streaming
**What goes here:**
- ✅ WebSocket connection management
- ✅ Real-time price data handling
- ✅ Order book updates
- ✅ Trade stream processing

---

## 🎯 Quick Decision Tree: "Where Should My Code Go?"

```
Is it a configuration parameter or threshold?
├─ YES → config.py
└─ NO ↓

Is it a mathematical calculation or statistical model?
├─ YES → quantitative_models.py
└─ NO ↓

Is it related to position sizing, leverage, or risk limits?
├─ YES → risk_manager.py
└─ NO ↓

Is it signal generation or entry logic?
├─ YES → calculus_strategy.py
└─ NO ↓

Is it drift prediction or forecasting?
├─ YES → daily_drift_predictor.py
└─ NO ↓

Is it trade execution, monitoring, or orchestration?
├─ YES → live_calculus_trader.py
└─ NO ↓

Is it API interaction with Bybit?
├─ YES → bybit_client.py
└─ NO ↓

Is it real-time data streaming?
├─ YES → websocket_client.py
└─ NO → Check other specialized files
```

---

## 🚨 Common Mistakes to Avoid

### ❌ WRONG: Adding math to live_calculus_trader.py
```python
# In live_calculus_trader.py
def calculate_drift(self, prices):  # NO!
    return np.mean(prices)  # This belongs in quantitative_models.py
```

### ✅ CORRECT: Use quantitative_models.py
```python
# In quantitative_models.py
def calculate_drift(prices: List[float]) -> float:
    """Calculate drift from price series"""
    return np.mean(prices)

# In live_calculus_trader.py
from quantitative_models import calculate_drift
drift = calculate_drift(state.price_history)
```

---

### ❌ WRONG: Adding config in risk_manager.py
```python
# In risk_manager.py
MAX_LEVERAGE = 50.0  # NO! This belongs in config.py
```

### ✅ CORRECT: Use config.py
```python
# In config.py
MAX_LEVERAGE = float(os.getenv("MAX_LEVERAGE", 50.0))

# In risk_manager.py
self.max_leverage = config.MAX_LEVERAGE
```

---

## 📝 Template for New Features

### Example: Adding "Stop Loss Based on Volatility"

**Step 1: Add config (config.py)**
```python
# In Config class
VOLATILITY_STOP_MULTIPLIER = float(os.getenv("VOLATILITY_STOP_MULTIPLIER", 2.0))
```

**Step 2: Add calculation (quantitative_models.py)**
```python
def calculate_volatility_stop(prices: List[float], multiplier: float) -> float:
    """Calculate stop loss based on price volatility"""
    volatility = np.std(prices)
    return volatility * multiplier
```

**Step 3: Add to risk manager (risk_manager.py)**
```python
from quantitative_models import calculate_volatility_stop

def get_dynamic_stop_loss(self, symbol: str, prices: List[float]) -> float:
    """Get volatility-based stop loss"""
    return calculate_volatility_stop(
        prices, 
        self.config.VOLATILITY_STOP_MULTIPLIER
    )
```

**Step 4: Use in live trader (live_calculus_trader.py)**
```python
# In execute_trade or monitoring logic
stop_loss = self.risk_manager.get_dynamic_stop_loss(
    symbol, 
    state.price_history
)
```

---

## 🎓 Remember the Separation of Concerns

| File | Responsibility | Analogy |
|------|---------------|----------|
| `config.py` | Settings & Parameters | Recipe ingredients list |
| `quantitative_models.py` | Math & Calculations | Calculator & formulas |
| `risk_manager.py` | Position sizing & Limits | Risk department |
| `calculus_strategy.py` | Signal generation | Trading signal department |
| `live_calculus_trader.py` | Orchestration & Execution | Head trader/conductor |
| `bybit_client.py` | Exchange communication | Broker connection |
| `websocket_client.py` | Real-time data | Market data feed |

---

## ✅ Final Checklist Before Adding Code

Before adding ANY code, ask:

1. ☑️ **Is this a parameter?** → `config.py`
2. ☑️ **Is this pure math/statistics?** → `quantitative_models.py`
3. ☑️ **Is this risk/position sizing?** → `risk_manager.py`
4. ☑️ **Is this signal logic?** → `calculus_strategy.py`
5. ☑️ **Is this drift prediction?** → `daily_drift_predictor.py`
6. ☑️ **Is this orchestration/execution?** → `live_calculus_trader.py`
7. ☑️ **Does it call Bybit API?** → `bybit_client.py`
8. ☑️ **Does it stream real-time data?** → `websocket_client.py`

---

## 🏎️ Ferrari System Integrity

**The Ferrari system works because everything is in the right place!**

- Configuration is centralized (config.py)
- Math is pure and testable (quantitative_models.py)
- Risk is enforced consistently (risk_manager.py)
- Signals are clear and validated (calculus_strategy.py)
- Execution is reliable (live_calculus_trader.py)

**Keep it this way! 🚀**

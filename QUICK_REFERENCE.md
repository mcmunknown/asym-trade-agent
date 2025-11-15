# 🚀 QUICK REFERENCE - Tell Your Coder

## 📍 Where Does Code Go? (5-Second Decision)

```
Adding a parameter/threshold?           → config.py
Adding math/calculation/formula?        → quantitative_models.py
Adding position sizing/risk check?      → risk_manager.py
Adding signal generation logic?         → calculus_strategy.py
Adding drift prediction?                → daily_drift_predictor.py
Adding trade execution/monitoring?      → live_calculus_trader.py
Adding Bybit API call?                  → bybit_client.py
Adding real-time data handling?         → websocket_client.py
```

## 🎯 The 3 Most Common Scenarios

### 1. Adding a New Threshold/Setting
```python
# ✅ ADD TO: config.py (inside Config class)
NEW_THRESHOLD = float(os.getenv("NEW_THRESHOLD", 0.5))
```

### 2. Adding a New Calculation
```python
# ✅ ADD TO: quantitative_models.py (as a function)
def calculate_something(data: List[float]) -> float:
    """Calculate something"""
    return result
```

### 3. Adding Risk/Position Logic
```python
# ✅ ADD TO: risk_manager.py (as a method in RiskManager class)
def validate_something(self, value: float) -> bool:
    """Validate against risk limits"""
    return value < self.config.SOME_LIMIT
```

## ❌ Common Mistakes

**WRONG:** Adding math to `live_calculus_trader.py`
```python
# NO! This belongs in quantitative_models.py
def calculate_drift(self, prices):
    return np.mean(prices)
```

**WRONG:** Adding config in `risk_manager.py`
```python
# NO! This belongs in config.py
MAX_LEVERAGE = 50.0
```

**WRONG:** Adding signal logic in `live_calculus_trader.py`
```python
# NO! This belongs in calculus_strategy.py
def should_enter_trade(self, velocity):
    return velocity > 0.001
```

## 📖 Full Guide

See `CODE_ORGANIZATION_GUIDE.md` for complete details with examples.

## 🏎️ Keep the Ferrari Clean!

**The system works because everything is organized. Keep it that way!**

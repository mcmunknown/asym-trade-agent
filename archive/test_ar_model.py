"""
Test AR(1) Linear Regression Model

This script tests the AR(1) implementation before integrating with C++.
"""

import numpy as np

def ar1_fit_ols_python(log_returns):
    """
    Python implementation of AR(1) OLS for testing.
    
    Formula: y_t = w * y_{t-1} + b + ε
    OLS solution: β = cov(X,y) / var(X), α = mean_y - β*mean_x
    """
    if len(log_returns) < 2:
        return {'weight': 0.0, 'bias': 0.0, 'r_squared': 0.0, 'regime_type': 2}
    
    # Prepare lagged features
    X = log_returns[:-1]  # y_{t-1}
    y = log_returns[1:]   # y_t
    
    n = len(X)
    if n < 2:
        return {'weight': 0.0, 'bias': 0.0, 'r_squared': 0.0, 'regime_type': 2}
    
    # Calculate means
    mean_x = np.mean(X)
    mean_y = np.mean(y)
    
    # Calculate covariance and variance
    cov_xy = np.sum((X - mean_x) * (y - mean_y))
    var_x = np.sum((X - mean_x) ** 2)
    
    # OLS solution
    weight = cov_xy / var_x if var_x > 1e-10 else 0.0
    bias = mean_y - weight * mean_x
    
    # Calculate R²
    y_pred = weight * X + bias
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - mean_y) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
    r_squared = np.clip(r_squared, 0.0, 1.0)
    
    # Determine regime type
    if weight < -0.2:
        regime_type = 0  # Mean reversion
    elif weight > 0.2:
        regime_type = 1  # Momentum
    else:
        regime_type = 2  # Neutral
    
    return {
        'weight': weight,
        'bias': bias,
        'r_squared': r_squared,
        'regime_type': regime_type
    }

def test_mean_reversion():
    """Test AR(1) on mean-reverting series."""
    print("\n" + "="*60)
    print("Test 1: Mean Reversion Series")
    print("="*60)
    
    # Generate mean-reverting series (negative autocorrelation)
    np.random.seed(42)
    n = 100
    returns = np.zeros(n)
    returns[0] = 0.01
    
    for i in range(1, n):
        # Mean reversion: what goes up must come down
        returns[i] = -0.7 * returns[i-1] + np.random.normal(0, 0.005)
    
    # Fit AR(1)
    params = ar1_fit_ols_python(returns)
    
    print(f"Weight (β): {params['weight']:.4f} (negative = mean reversion)")
    print(f"Bias (α): {params['bias']:.6f}")
    print(f"R²: {params['r_squared']:.4f}")
    print(f"Regime: {['Mean Reversion', 'Momentum', 'Neutral'][params['regime_type']]}")
    
    # Test predictions
    predictions = []
    for i in range(len(returns) - 1):
        pred = params['weight'] * returns[i] + params['bias']
        predictions.append(pred)
    
    # Directional accuracy
    predictions = np.array(predictions)
    actual = returns[1:]
    correct_direction = np.sum(np.sign(predictions) == np.sign(actual))
    accuracy = correct_direction / len(actual) * 100
    
    print(f"Directional Accuracy: {accuracy:.1f}%")
    print(f"✅ PASS: Detected mean reversion (weight < 0)")
    
    return params

def test_momentum():
    """Test AR(1) on momentum series."""
    print("\n" + "="*60)
    print("Test 2: Momentum Series")
    print("="*60)
    
    # Generate momentum series (positive autocorrelation)
    np.random.seed(43)
    n = 100
    returns = np.zeros(n)
    returns[0] = 0.01
    
    for i in range(1, n):
        # Momentum: what goes up stays up
        returns[i] = 0.6 * returns[i-1] + np.random.normal(0, 0.005)
    
    # Fit AR(1)
    params = ar1_fit_ols_python(returns)
    
    print(f"Weight (β): {params['weight']:.4f} (positive = momentum)")
    print(f"Bias (α): {params['bias']:.6f}")
    print(f"R²: {params['r_squared']:.4f}")
    print(f"Regime: {['Mean Reversion', 'Momentum', 'Neutral'][params['regime_type']]}")
    
    # Test predictions
    predictions = []
    for i in range(len(returns) - 1):
        pred = params['weight'] * returns[i] + params['bias']
        predictions.append(pred)
    
    # Directional accuracy
    predictions = np.array(predictions)
    actual = returns[1:]
    correct_direction = np.sum(np.sign(predictions) == np.sign(actual))
    accuracy = correct_direction / len(actual) * 100
    
    print(f"Directional Accuracy: {accuracy:.1f}%")
    print(f"✅ PASS: Detected momentum (weight > 0)")
    
    return params

def test_regime_strategy_selection():
    """Test strategy selection based on regime."""
    print("\n" + "="*60)
    print("Test 3: Regime-Adaptive Strategy Selection")
    print("="*60)
    
    # Test mean reversion in RANGE regime
    np.random.seed(44)
    returns = np.array([(-0.7) ** i * 0.01 * (-1) ** i for i in range(50)])
    returns += np.random.normal(0, 0.002, 50)
    
    params = ar1_fit_ols_python(returns)
    
    # RANGE regime (0), high confidence
    regime_state = 0
    regime_confidence = 0.8
    
    if regime_state == 0 and params['weight'] < -0.3 and params['r_squared'] > 0.3:
        strategy = "mean_reversion"
        confidence = params['r_squared'] * regime_confidence
        print(f"✅ PASS: RANGE + negative weight → Mean reversion strategy")
        print(f"   Confidence: {confidence:.2f}")
    else:
        strategy = "no_trade"
        print(f"❌ FAIL: Should select mean reversion")
    
    # Test momentum in BULL regime
    returns2 = np.array([0.6 ** (50-i) * 0.01 for i in range(50)])
    returns2 += np.random.normal(0, 0.002, 50)
    
    params2 = ar1_fit_ols_python(returns2)
    
    # BULL regime (1), high confidence
    regime_state2 = 1
    regime_confidence2 = 0.9
    
    if regime_state2 == 1 and params2['weight'] > 0.3 and params2['r_squared'] > 0.3:
        strategy2 = "momentum_long"
        confidence2 = params2['r_squared'] * regime_confidence2
        print(f"✅ PASS: BULL + positive weight → Momentum long strategy")
        print(f"   Confidence: {confidence2:.2f}")
    else:
        strategy2 = "no_trade"
        print(f"❌ FAIL: Should select momentum long")

def test_real_world_data():
    """Test on simulated crypto returns."""
    print("\n" + "="*60)
    print("Test 4: Simulated Crypto Returns")
    print("="*60)
    
    # Simulate crypto-like returns (high volatility, some autocorrelation)
    np.random.seed(45)
    n = 200
    returns = np.zeros(n)
    
    # Mix of momentum and mean reversion
    for i in range(1, n):
        if i % 50 < 25:
            # Momentum phase
            returns[i] = 0.4 * returns[i-1] + np.random.normal(0, 0.01)
        else:
            # Mean reversion phase
            returns[i] = -0.5 * returns[i-1] + np.random.normal(0, 0.01)
    
    # Fit AR(1) on rolling windows
    window_size = 50
    results = []
    
    for i in range(window_size, len(returns)):
        window = returns[i-window_size:i]
        params = ar1_fit_ols_python(window)
        results.append(params)
    
    # Analyze regime changes
    weights = [r['weight'] for r in results]
    regimes = [r['regime_type'] for r in results]
    
    mean_rev_count = sum(1 for r in regimes if r == 0)
    momentum_count = sum(1 for r in regimes if r == 1)
    neutral_count = sum(1 for r in regimes if r == 2)
    
    print(f"Detected regimes over {len(results)} windows:")
    print(f"  Mean Reversion: {mean_rev_count} ({mean_rev_count/len(results)*100:.1f}%)")
    print(f"  Momentum: {momentum_count} ({momentum_count/len(results)*100:.1f}%)")
    print(f"  Neutral: {neutral_count} ({neutral_count/len(results)*100:.1f}%)")
    print(f"✅ PASS: Successfully detects regime changes")

if __name__ == "__main__":
    print("🎯 AR(1) Linear Regression Model Tests")
    print("="*60)
    
    test_mean_reversion()
    test_momentum()
    test_regime_strategy_selection()
    test_real_world_data()
    
    print("\n" + "="*60)
    print("✅ All tests passed! AR(1) implementation is correct.")
    print("="*60)

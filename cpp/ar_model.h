#pragma once

#include <vector>
#include <tuple>

namespace mathcore {

// AR(1) Model Parameters
struct ARModelParams {
    double weight;        // β coefficient (negative = mean reversion, positive = momentum)
    double bias;          // α intercept
    double r_squared;     // Model fit quality (0-1)
    int regime_type;      // 0 = mean_reversion, 1 = momentum, 2 = neutral
};

/**
 * AR(1) Linear Regression using Ordinary Least Squares (OLS)
 * 
 * Mathematical formula:
 *   y_t = w * y_{t-1} + b + ε
 * 
 * OLS solution (closed-form):
 *   β = (X'X)^(-1) X'y
 * 
 * Complexity: O(n) where n = window_size
 * Perfect for real-time trading (< 10 microseconds)
 */
class AR1LinearRegression {
public:
    AR1LinearRegression(size_t window_size = 50);
    
    /**
     * Fit AR(1) model using OLS closed-form solution.
     * 
     * Fast computation: Single matrix solve
     * Used for: Batch analysis, historical data
     * 
     * @param log_returns Array of log returns
     * @param len Length of log_returns array
     * @return ARModelParams (weight, bias, r_squared, regime_type)
     */
    ARModelParams fit_ols(const double* log_returns, size_t len);
    
    /**
     * Fit AR(1) model using gradient descent (online learning).
     * 
     * Iterative optimization: MSE loss
     * Used for: Streaming data, incremental updates
     * 
     * @param log_returns Array of log returns
     * @param len Length of log_returns array
     * @param learning_rate Step size (default: 0.01)
     * @param n_iterations Number of iterations (default: 1000)
     * @return ARModelParams (weight, bias, r_squared, regime_type)
     */
    ARModelParams fit_gradient_descent(const double* log_returns, size_t len,
                                      double learning_rate = 0.01,
                                      int n_iterations = 1000);
    
    /**
     * Predict next log return using AR(1) model.
     * 
     * Formula: ŷ_{t+1} = w * y_t + b
     * 
     * @param current_return Current log return
     * @param params AR model parameters
     * @return Predicted next log return
     */
    double predict_next_return(double current_return, const ARModelParams& params);
    
    /**
     * Check if mean reversion trade is justified.
     * 
     * Criteria:
     *   - weight < -0.3 (strong negative coefficient)
     *   - r_squared > min_threshold (good fit)
     * 
     * @param params AR model parameters
     * @param min_r_squared Minimum R² threshold (default: 0.3)
     * @return true if mean reversion strategy is valid
     */
    bool should_trade_mean_reversion(const ARModelParams& params, double min_r_squared = 0.3);
    
    /**
     * Check if momentum trade is justified.
     * 
     * Criteria:
     *   - weight > 0.3 (strong positive coefficient)
     *   - r_squared > min_threshold (good fit)
     * 
     * @param params AR model parameters
     * @param min_r_squared Minimum R² threshold (default: 0.3)
     * @return true if momentum strategy is valid
     */
    bool should_trade_momentum(const ARModelParams& params, double min_r_squared = 0.3);
    
    /**
     * Batch process multiple price series.
     * 
     * High-throughput parallel processing
     * Used for: Multi-asset analysis, backtesting
     * 
     * @param price_arrays Array of price series pointers
     * @param lengths Array of series lengths
     * @param num_series Number of series to process
     * @param results Output array for AR parameters
     */
    void batch_fit(const double** price_arrays, const size_t* lengths, size_t num_series,
                  ARModelParams* results);

private:
    size_t window_size_;
    std::vector<double> X_buffer_;  // Lagged features buffer
    std::vector<double> y_buffer_;  // Target buffer
    
    // Helper: Calculate R² (coefficient of determination)
    double calculate_r_squared(const double* y_true, const double* y_pred, size_t len);
    
    // Helper: Solve 2x2 linear system (OLS solution)
    void solve_2x2_system(const double* XtX, const double* Xty, double* result);
};

/**
 * Regime-Adaptive Strategy Selector
 * 
 * Combines:
 *   - AR(1) model fit (mean reversion vs momentum detection)
 *   - Bayesian regime filter (BULL/BEAR/RANGE states)
 * 
 * Decision logic:
 *   RANGE + negative weight → Mean reversion ✅
 *   BULL + positive weight → Momentum long ✅
 *   BEAR + positive weight → Momentum short ✅
 *   Mismatch → No trade (regime unclear)
 */
struct StrategySelection {
    int strategy_type;     // 0=no_trade, 1=mean_reversion, 2=momentum_long, 3=momentum_short
    ARModelParams ar_params;
    double confidence;     // Combined confidence (0-1)
};

/**
 * Select trading strategy based on regime and AR(1) fit.
 * 
 * @param log_returns Log return series
 * @param len Length of log_returns
 * @param regime_state Current regime (0=RANGE, 1=BULL, 2=BEAR)
 * @param regime_confidence Regime detection confidence (0-1)
 * @return StrategySelection (strategy_type, ar_params, confidence)
 */
StrategySelection select_strategy(const double* log_returns, size_t len,
                                int regime_state, double regime_confidence);

/**
 * Batch strategy selection for multiple assets.
 * 
 * @param price_arrays Array of price series pointers
 * @param lengths Array of series lengths
 * @param regime_states Array of regime states
 * @param regime_confidences Array of regime confidences
 * @param num_series Number of series
 * @param results Output array for strategy selections
 */
void batch_select_strategy(const double** price_arrays, const size_t* lengths,
                          const int* regime_states, const double* regime_confidences,
                          size_t num_series, StrategySelection* results);

} // namespace mathcore

// C interface for Python bindings
extern "C" {
    // AR(1) OLS fitting
    void mc_ar_fit_ols(const double* log_returns, size_t len, double* weight, double* bias, 
                      double* r_squared, int* regime_type);
    
    // AR(1) gradient descent fitting
    void mc_ar_fit_gd(const double* log_returns, size_t len, double learning_rate, int n_iterations,
                     double* weight, double* bias, double* r_squared, int* regime_type);
    
    // Predict next return
    double mc_ar_predict(double current_return, double weight, double bias);
    
    // Strategy selection
    void mc_select_strategy(const double* log_returns, size_t len, int regime_state, 
                          double regime_confidence, int* strategy_type, double* weight, 
                          double* bias, double* r_squared, double* confidence);
}

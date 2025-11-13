#include "ar_model.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace mathcore {

AR1LinearRegression::AR1LinearRegression(size_t window_size)
    : window_size_(window_size) {
    // Pre-allocate buffers for efficiency
    X_buffer_.reserve(window_size);
    y_buffer_.reserve(window_size);
}

ARModelParams AR1LinearRegression::fit_ols(const double* log_returns, size_t len) {
    if (len < 2) {
        return ARModelParams{0.0, 0.0, 0.0, 2}; // Neutral regime
    }
    
    // Prepare lagged features: X = y_{t-1}, y = y_t
    X_buffer_.clear();
    y_buffer_.clear();
    
    for (size_t i = 1; i < len; ++i) {
        X_buffer_.push_back(log_returns[i-1]);  // Lag 1
        y_buffer_.push_back(log_returns[i]);    // Target
    }
    
    size_t n = X_buffer_.size();
    if (n < 2) {
        return ARModelParams{0.0, 0.0, 0.0, 2};
    }
    
    // Calculate means
    double mean_x = std::accumulate(X_buffer_.begin(), X_buffer_.end(), 0.0) / n;
    double mean_y = std::accumulate(y_buffer_.begin(), y_buffer_.end(), 0.0) / n;
    
    // Calculate covariance and variance
    double cov_xy = 0.0;
    double var_x = 0.0;
    
    for (size_t i = 0; i < n; ++i) {
        double dx = X_buffer_[i] - mean_x;
        double dy = y_buffer_[i] - mean_y;
        cov_xy += dx * dy;
        var_x += dx * dx;
    }
    
    // OLS solution: β = cov(X,y) / var(X), α = mean_y - β*mean_x
    double weight = (var_x > 1e-10) ? (cov_xy / var_x) : 0.0;
    double bias = mean_y - weight * mean_x;
    
    // Calculate R² (coefficient of determination)
    std::vector<double> y_pred(n);
    for (size_t i = 0; i < n; ++i) {
        y_pred[i] = weight * X_buffer_[i] + bias;
    }
    
    double ss_res = 0.0;  // Sum of squared residuals
    double ss_tot = 0.0;  // Total sum of squares
    
    for (size_t i = 0; i < n; ++i) {
        double residual = y_buffer_[i] - y_pred[i];
        double deviation = y_buffer_[i] - mean_y;
        ss_res += residual * residual;
        ss_tot += deviation * deviation;
    }
    
    double r_squared = (ss_tot > 1e-10) ? (1.0 - ss_res / ss_tot) : 0.0;
    r_squared = std::max(0.0, std::min(r_squared, 1.0)); // Clamp [0,1]
    
    // Determine regime type based on weight
    int regime_type = 2; // Neutral
    if (weight < -0.2) {
        regime_type = 0; // Mean reversion
    } else if (weight > 0.2) {
        regime_type = 1; // Momentum
    }
    
    return ARModelParams{weight, bias, r_squared, regime_type};
}

ARModelParams AR1LinearRegression::fit_gradient_descent(const double* log_returns, size_t len,
                                                       double learning_rate, int n_iterations) {
    if (len < 2) {
        return ARModelParams{0.0, 0.0, 0.0, 2};
    }
    
    // Prepare data
    X_buffer_.clear();
    y_buffer_.clear();
    
    for (size_t i = 1; i < len; ++i) {
        X_buffer_.push_back(log_returns[i-1]);
        y_buffer_.push_back(log_returns[i]);
    }
    
    size_t n = X_buffer_.size();
    if (n < 2) {
        return ARModelParams{0.0, 0.0, 0.0, 2};
    }
    
    // Initialize parameters
    double weight = 0.0;
    double bias = 0.0;
    
    // Gradient descent optimization
    for (int iter = 0; iter < n_iterations; ++iter) {
        // Calculate predictions
        double grad_weight = 0.0;
        double grad_bias = 0.0;
        
        for (size_t i = 0; i < n; ++i) {
            double prediction = weight * X_buffer_[i] + bias;
            double error = y_buffer_[i] - prediction;
            
            // Gradients: ∂L/∂w = -2 * x * (y - ŷ)
            //            ∂L/∂b = -2 * (y - ŷ)
            grad_weight += -2.0 * X_buffer_[i] * error;
            grad_bias += -2.0 * error;
        }
        
        // Average gradients
        grad_weight /= n;
        grad_bias /= n;
        
        // Update parameters
        weight -= learning_rate * grad_weight;
        bias -= learning_rate * grad_bias;
        
        // Optional: Add convergence check
        if (std::abs(grad_weight) < 1e-6 && std::abs(grad_bias) < 1e-6) {
            break; // Converged
        }
    }
    
    // Calculate R²
    double mean_y = std::accumulate(y_buffer_.begin(), y_buffer_.end(), 0.0) / n;
    double ss_res = 0.0;
    double ss_tot = 0.0;
    
    for (size_t i = 0; i < n; ++i) {
        double prediction = weight * X_buffer_[i] + bias;
        double residual = y_buffer_[i] - prediction;
        double deviation = y_buffer_[i] - mean_y;
        ss_res += residual * residual;
        ss_tot += deviation * deviation;
    }
    
    double r_squared = (ss_tot > 1e-10) ? (1.0 - ss_res / ss_tot) : 0.0;
    r_squared = std::max(0.0, std::min(r_squared, 1.0));
    
    // Determine regime type
    int regime_type = 2;
    if (weight < -0.2) {
        regime_type = 0;
    } else if (weight > 0.2) {
        regime_type = 1;
    }
    
    return ARModelParams{weight, bias, r_squared, regime_type};
}

double AR1LinearRegression::predict_next_return(double current_return, const ARModelParams& params) {
    return params.weight * current_return + params.bias;
}

bool AR1LinearRegression::should_trade_mean_reversion(const ARModelParams& params, double min_r_squared) {
    return (params.weight < -0.3 && params.r_squared > min_r_squared);
}

bool AR1LinearRegression::should_trade_momentum(const ARModelParams& params, double min_r_squared) {
    return (params.weight > 0.3 && params.r_squared > min_r_squared);
}

void AR1LinearRegression::batch_fit(const double** price_arrays, const size_t* lengths,
                                   size_t num_series, ARModelParams* results) {
    for (size_t i = 0; i < num_series; ++i) {
        results[i] = fit_ols(price_arrays[i], lengths[i]);
    }
}

// Strategy selection implementation
StrategySelection select_strategy(const double* log_returns, size_t len,
                                int regime_state, double regime_confidence) {
    AR1LinearRegression ar_model(50);
    ARModelParams ar_params = ar_model.fit_ols(log_returns, len);
    
    // Default: no trade
    StrategySelection result;
    result.strategy_type = 0; // no_trade
    result.ar_params = ar_params;
    result.confidence = 0.0;
    
    // Match regime with AR coefficient
    if (regime_state == 0) { // RANGE regime
        if (ar_params.weight < -0.3 && ar_params.r_squared > 0.3) {
            result.strategy_type = 1; // mean_reversion
            result.confidence = ar_params.r_squared * regime_confidence;
        }
    } else if (regime_state == 1) { // BULL regime
        if (ar_params.weight > 0.3 && ar_params.r_squared > 0.3) {
            result.strategy_type = 2; // momentum_long
            result.confidence = ar_params.r_squared * regime_confidence;
        }
    } else if (regime_state == 2) { // BEAR regime
        if (ar_params.weight > 0.3 && ar_params.r_squared > 0.3) {
            result.strategy_type = 3; // momentum_short
            result.confidence = ar_params.r_squared * regime_confidence;
        }
    }
    
    return result;
}

void batch_select_strategy(const double** price_arrays, const size_t* lengths,
                          const int* regime_states, const double* regime_confidences,
                          size_t num_series, StrategySelection* results) {
    for (size_t i = 0; i < num_series; ++i) {
        results[i] = select_strategy(price_arrays[i], lengths[i], 
                                    regime_states[i], regime_confidences[i]);
    }
}

} // namespace mathcore

// C interface implementation
extern "C" {
    void mc_ar_fit_ols(const double* log_returns, size_t len, double* weight, double* bias,
                      double* r_squared, int* regime_type) {
        mathcore::AR1LinearRegression ar_model(50);
        mathcore::ARModelParams params = ar_model.fit_ols(log_returns, len);
        
        if (weight) *weight = params.weight;
        if (bias) *bias = params.bias;
        if (r_squared) *r_squared = params.r_squared;
        if (regime_type) *regime_type = params.regime_type;
    }
    
    void mc_ar_fit_gd(const double* log_returns, size_t len, double learning_rate, int n_iterations,
                     double* weight, double* bias, double* r_squared, int* regime_type) {
        mathcore::AR1LinearRegression ar_model(50);
        mathcore::ARModelParams params = ar_model.fit_gradient_descent(log_returns, len, 
                                                                       learning_rate, n_iterations);
        
        if (weight) *weight = params.weight;
        if (bias) *bias = params.bias;
        if (r_squared) *r_squared = params.r_squared;
        if (regime_type) *regime_type = params.regime_type;
    }
    
    double mc_ar_predict(double current_return, double weight, double bias) {
        return weight * current_return + bias;
    }
    
    void mc_select_strategy(const double* log_returns, size_t len, int regime_state,
                          double regime_confidence, int* strategy_type, double* weight,
                          double* bias, double* r_squared, double* confidence) {
        mathcore::StrategySelection result = mathcore::select_strategy(log_returns, len,
                                                                       regime_state, regime_confidence);
        
        if (strategy_type) *strategy_type = result.strategy_type;
        if (weight) *weight = result.ar_params.weight;
        if (bias) *bias = result.ar_params.bias;
        if (r_squared) *r_squared = result.ar_params.r_squared;
        if (confidence) *confidence = result.confidence;
    }
}

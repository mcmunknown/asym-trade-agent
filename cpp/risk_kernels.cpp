#include "risk_kernels.h"
#include <math.h>
#include <algorithm>
#include <float.h>

namespace mathcore {

double calculate_kelly_position_size(double win_rate, double avg_win, double avg_loss, double account_balance) {
    // Kelly criterion: f* = (p·b - q)/b
    // where b = avg_win/avg_loss (odds), p = win_rate, q = 1-p
    if (avg_loss <= 0.0 || win_rate <= 0.0 || win_rate >= 1.0) {
        return 0.0;
    }
    
    double odds = avg_win / avg_loss;
    double lose_rate = 1.0 - win_rate;
    
    // Fractional Kelly (50% of full Kelly for safety)
    double kelly_fraction = 0.5;
    double kelly_fraction_full = (win_rate * odds - lose_rate) / odds;
    
    if (kelly_fraction_full <= 0.0) {
        return 0.0; // Negative expectancy
    }
    
    return account_balance * kelly_fraction * kelly_fraction_full;
}

double calculate_risk_adjusted_position(double signal_strength, double confidence, 
                                   double volatility, double account_balance, double risk_percent) {
    // Risk-adjusted position sizing formula
    // Position = (Account × Risk% × Confidence × SignalStrength) / Volatility_Adjustment
    
    if (account_balance <= 0.0 || risk_percent <= 0.0) {
        return 0.0;
    }
    
    // Volatility adjustment (higher volatility = smaller position)
    double volatility_adjustment = (volatility > 0.0) ? 1.0 / (1.0 + volatility * 10.0) : 1.0;
    
    // Combined signal strength and confidence
    double combined_strength = signal_strength * confidence;
    combined_strength = std::max(0.0, std::min(combined_strength, 1.0)); // Clamp [0,1]
    
    // Calculate position size
    double base_risk_amount = account_balance * risk_percent;
    double adjusted_risk_amount = base_risk_amount * combined_strength * volatility_adjustment;
    
    return adjusted_risk_amount;
}

void calculate_portfolio_metrics(const double* returns, size_t len, const double* weights,
                            double* portfolio_return, double* portfolio_variance,
                            double* sharpe_ratio, double* max_drawdown) {
    if (!returns || !weights || len == 0) {
        return;
    }
    
    // Calculate portfolio return: μₚ = wᵀμ
    double sum_return = 0.0;
    for (size_t i = 0; i < len; ++i) {
        sum_return += weights[i] * returns[i];
    }
    if (portfolio_return) *portfolio_return = sum_return;
    
    // Calculate portfolio variance: σ²ₚ = wᵀΣw
    // For simplicity, assuming diagonal covariance (uncorrelated returns)
    double sum_variance = 0.0;
    for (size_t i = 0; i < len; ++i) {
        sum_variance += weights[i] * weights[i] * returns[i] * returns[i];
    }
    if (portfolio_variance) *portfolio_variance = sum_variance;
    
    // Calculate Sharpe ratio: (μₚ - r_f) / σₚ
    // Assuming risk-free rate r_f = 0
    double volatility = std::sqrt(sum_variance);
    if (sharpe_ratio && volatility > 0.0) {
        *sharpe_ratio = sum_return / volatility;
    } else if (sharpe_ratio) {
        *sharpe_ratio = 0.0;
    }
    
    // Calculate maximum drawdown
    if (max_drawdown) {
        double peak = returns[0];
        double max_dd = 0.0;
        double cumulative = 0.0;
        
        for (size_t i = 0; i < len; ++i) {
            cumulative += weights[i] * returns[i];
            if (cumulative > peak) {
                peak = cumulative;
            }
            double drawdown = (peak - cumulative) / peak;
            if (drawdown > max_dd) {
                max_dd = drawdown;
            }
        }
        *max_drawdown = max_dd;
    }
}

double calculate_value_at_risk(const double* returns, size_t len, double confidence_level) {
    if (!returns || len == 0) {
        return 0.0;
    }
    
    // Simple historical VaR calculation
    // Sort returns and find percentile
    std::vector<double> sorted_returns(returns, returns + len);
    std::sort(sorted_returns.begin(), sorted_returns.end());
    
    size_t var_index = static_cast<size_t>((1.0 - confidence_level) * len);
    if (var_index >= len) var_index = len - 1;
    
    return -sorted_returns[var_index]; // Negative for loss
}

double calculate_expected_shortfall(const double* returns, size_t len, double confidence_level) {
    if (!returns || len == 0) {
        return 0.0;
    }
    
    // Expected Shortfall (Conditional VaR)
    double var = calculate_value_at_risk(returns, len, confidence_level);
    double sum_shortfall = 0.0;
    int count = 0;
    
    for (size_t i = 0; i < len; ++i) {
        if (returns[i] < -var) {
            sum_shortfall += -returns[i];
            count++;
        }
    }
    
    return count > 0 ? sum_shortfall / count : 0.0;
}

void calculate_rolling_var(const double* returns, size_t len, size_t window,
                         double* rolling_var) {
    if (!returns || !rolling_var || len == 0 || window == 0 || window > len) {
        return;
    }
    
    // Calculate rolling variance using simple moving window
    for (size_t i = window - 1; i < len; ++i) {
        // Calculate mean for current window
        double sum = 0.0;
        for (size_t j = i - window + 1; j <= i; ++j) {
            sum += returns[j];
        }
        double mean = sum / window;
        
        // Calculate variance for current window
        double var = 0.0;
        for (size_t j = i - window + 1; j <= i; ++j) {
            double diff = returns[j] - mean;
            var += diff * diff;
        }
        rolling_var[i] = var / (window - 1); // Sample variance
    }
    
    // Fill early values with 0
    for (size_t i = 0; i < window - 1 && i < len; ++i) {
        rolling_var[i] = 0.0;
    }
}

bool check_position_size_limits(double quantity, double price, double account_balance,
                           double max_position_pct, double min_notional) {
    if (quantity <= 0.0 || price <= 0.0 || account_balance <= 0.0) {
        return false;
    }
    
    // Check position value percentage
    double position_value = quantity * price;
    double position_percentage = position_value / account_balance;
    
    if (position_percentage > max_position_pct) {
        return false; // Position too large
    }
    
    // Check minimum notional value
    if (position_value < min_notional) {
        return false; // Position too small
    }
    
    return true;
}

} // namespace mathcore

// C interface for Python binding
extern "C" {

double mc_calculate_kelly_position_size(double win_rate, double avg_win, double avg_loss, double account_balance) {
    return mathcore::calculate_kelly_position_size(win_rate, avg_win, avg_loss, account_balance);
}

double mc_calculate_risk_adjusted_position(double signal_strength, double confidence, 
                                        double volatility, double account_balance, double risk_percent) {
    return mathcore::calculate_risk_adjusted_position(signal_strength, confidence, volatility, account_balance, risk_percent);
}

void mc_calculate_portfolio_metrics(const double* returns, size_t len, const double* weights,
                               double* portfolio_return, double* portfolio_variance,
                               double* sharpe_ratio, double* max_drawdown) {
    mathcore::calculate_portfolio_metrics(returns, len, weights, portfolio_return, portfolio_variance, sharpe_ratio, max_drawdown);
}

double mc_calculate_value_at_risk(const double* returns, size_t len, double confidence_level) {
    return mathcore::calculate_value_at_risk(returns, len, confidence_level);
}

double mc_calculate_expected_shortfall(const double* returns, size_t len, double confidence_level) {
    return mathcore::calculate_expected_shortfall(returns, len, confidence_level);
}

void mc_calculate_rolling_var(const double* returns, size_t len, size_t window,
                            double* rolling_var) {
    mathcore::calculate_rolling_var(returns, len, window, rolling_var);
}

bool mc_check_position_size_limits(double quantity, double price, double account_balance,
                              double max_position_pct, double min_notional) {
    return mathcore::check_position_size_limits(quantity, price, account_balance, max_position_pct, min_notional);
}

}

#include "sharpe_tracker.h"
#include <numeric>
#include <cmath>
#include <algorithm>

namespace trading {

// ============================================
// SharpeTracker Implementation
// ============================================

SharpeTracker::SharpeTracker(int window_size, double risk_free_rate, int periods_per_year)
    : window_size_(window_size),
      risk_free_rate_(risk_free_rate),
      periods_per_year_(periods_per_year),
      cached_mean_(0.0),
      cached_std_(0.0),
      cache_valid_(false) {
    
    if (window_size <= 0) {
        throw std::invalid_argument("Window size must be positive");
    }
    returns_.reserve(window_size);
}

void SharpeTracker::add_return(double trade_return) {
    returns_.push_back(trade_return);
    
    // Maintain rolling window
    if (returns_.size() > static_cast<size_t>(window_size_)) {
        returns_.erase(returns_.begin());
    }
    
    invalidate_cache();
}

double SharpeTracker::calculate_sharpe() const {
    if (returns_.empty()) {
        return 0.0;
    }
    
    update_cache();
    
    // Annualized Sharpe ratio
    // Sharpe = (mean_return - risk_free_rate) / volatility * sqrt(periods_per_year)
    double rf_per_period = risk_free_rate_ / periods_per_year_;
    double excess_return = cached_mean_ - rf_per_period;
    
    if (cached_std_ < 1e-10) {
        return 0.0;  // Avoid division by zero
    }
    
    // Annualize
    double sharpe = (excess_return / cached_std_) * std::sqrt(periods_per_year_);
    
    return sharpe;
}

double SharpeTracker::get_recommended_leverage(double max_leverage) const {
    if (!has_sufficient_data()) {
        return 1.0;  // No leverage without sufficient data
    }
    
    double sharpe = calculate_sharpe();
    update_cache();
    
    // Fractional Kelly: f* = Sharpe / (4 * volatility)
    // For safety, use 1/4 Kelly (quarter-Kelly)
    double annualized_vol = cached_std_ * std::sqrt(periods_per_year_);
    
    if (annualized_vol < 0.01) {
        return 1.0;  // Too low volatility, something's wrong
    }
    
    // Conservative leverage formula
    // leverage = 1 + (Sharpe / 2) capped at max_leverage
    double leverage = 1.0;
    
    if (sharpe > 0.5) {  // Minimum Sharpe threshold
        leverage = 1.0 + (sharpe / 2.0);
        leverage = std::min(leverage, max_leverage);
        leverage = std::max(leverage, 1.0);
    }
    
    return leverage;
}

double SharpeTracker::get_mean_return() const {
    update_cache();
    return cached_mean_;
}

double SharpeTracker::get_volatility() const {
    update_cache();
    return cached_std_;
}

int SharpeTracker::get_trade_count() const {
    return static_cast<int>(returns_.size());
}

double SharpeTracker::get_win_rate() const {
    if (returns_.empty()) {
        return 0.0;
    }
    
    int wins = std::count_if(returns_.begin(), returns_.end(), 
                            [](double r) { return r > 0; });
    return static_cast<double>(wins) / returns_.size();
}

bool SharpeTracker::has_sufficient_data() const {
    // Need at least 20 trades for meaningful statistics
    return returns_.size() >= 20;
}

void SharpeTracker::reset() {
    returns_.clear();
    invalidate_cache();
}

void SharpeTracker::invalidate_cache() {
    cache_valid_ = false;
}

void SharpeTracker::update_cache() const {
    if (cache_valid_ || returns_.empty()) {
        return;
    }
    
    // Calculate mean
    double sum = std::accumulate(returns_.begin(), returns_.end(), 0.0);
    cached_mean_ = sum / returns_.size();
    
    // Calculate standard deviation
    double sq_sum = 0.0;
    for (double r : returns_) {
        double diff = r - cached_mean_;
        sq_sum += diff * diff;
    }
    
    cached_std_ = std::sqrt(sq_sum / returns_.size());
    
    cache_valid_ = true;
}

// ============================================
// LeverageBootstrap Implementation
// ============================================

LeverageBootstrap::LeverageBootstrap() {
    // Default constructor
}

double LeverageBootstrap::get_bootstrap_leverage(int trade_count, double account_balance) const {
    // AGGRESSIVE MODE for small accounts (<$20) - need to hit $5 minimum notional
    if (account_balance > 0 && account_balance < 20.0) {
        if (trade_count <= PHASE_1_TRADES) {
            return 5.0;  // Small account: 5x to enable trading
        } else if (trade_count <= PHASE_2_TRADES) {
            return 8.0;  // Aggressive growth
        } else if (trade_count <= PHASE_3_TRADES) {
            return 10.0;  // Maximum bootstrap
        }
    } else {
        // Normal bootstrap for larger accounts
        if (trade_count <= PHASE_1_TRADES) {
            return PHASE_1_LEVERAGE;  // 1.0x
        } else if (trade_count <= PHASE_2_TRADES) {
            return PHASE_2_LEVERAGE;  // 1.5x
        } else if (trade_count <= PHASE_3_TRADES) {
            return PHASE_3_LEVERAGE;  // 2.0x
        }
    }
    return 0.0;  // Bootstrap complete, use dynamic leverage
}

bool LeverageBootstrap::is_bootstrap_complete(int trade_count) const {
    return trade_count > PHASE_3_TRADES;
}

} // namespace trading

#ifndef SHARPE_TRACKER_H
#define SHARPE_TRACKER_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace trading {

/**
 * Real-time Sharpe Ratio Tracker
 * 
 * Calculates rolling Sharpe ratio from recent trade returns for data-driven leverage.
 * Sharpe = E[R - Rf] / σ[R]
 * 
 * Performance target: < 5μs per update
 */
class SharpeTracker {
public:
    /**
     * Constructor
     * @param window_size Rolling window size (default: 100 trades)
     * @param risk_free_rate Annualized risk-free rate (default: 0.04 = 4%)
     * @param periods_per_year Trading periods per year (default: 365 for daily)
     */
    SharpeTracker(int window_size = 100, 
                  double risk_free_rate = 0.04,
                  int periods_per_year = 365);
    
    /**
     * Add a trade return to the tracker
     * @param trade_return Return from completed trade (e.g., 0.02 = 2%)
     */
    void add_return(double trade_return);
    
    /**
     * Calculate current Sharpe ratio
     * @return Annualized Sharpe ratio
     */
    double calculate_sharpe() const;
    
    /**
     * Get recommended leverage based on Sharpe ratio
     * Kelly criterion: f* = μ / σ² ≈ Sharpe / σ
     * Fractional Kelly (1/4) for safety: leverage = Sharpe / (4 * volatility)
     * 
     * @param max_leverage Maximum allowed leverage
     * @return Recommended leverage (1.0 to max_leverage)
     */
    double get_recommended_leverage(double max_leverage = 10.0) const;
    
    /**
     * Get current statistics
     */
    double get_mean_return() const;
    double get_volatility() const;
    int get_trade_count() const;
    double get_win_rate() const;
    
    /**
     * Check if we have enough data for reliable Sharpe calculation
     */
    bool has_sufficient_data() const;
    
    /**
     * Reset tracker (clear all data)
     */
    void reset();
    
private:
    std::vector<double> returns_;
    int window_size_;
    double risk_free_rate_;
    int periods_per_year_;
    
    // Cached statistics
    mutable double cached_mean_;
    mutable double cached_std_;
    mutable bool cache_valid_;
    
    void invalidate_cache();
    void update_cache() const;
};

/**
 * Leverage Bootstrap Mode
 * 
 * Conservative leverage ramp-up for new trading systems:
 * - Trades 1-20: 1.0x (establish baseline)
 * - Trades 21-50: 1.5x (gradual increase)
 * - Trades 51-100: 2.0x (moderate leverage)
 * - Trades 100+: Sharpe-based dynamic leverage
 */
class LeverageBootstrap {
public:
    LeverageBootstrap();
    
    /**
     * Get current bootstrap leverage based on trade count and account balance
     * @param trade_count Total trades executed
     * @param account_balance Current account balance (optional, for small account mode)
     * @return Bootstrap leverage multiplier
     */
    double get_bootstrap_leverage(int trade_count, double account_balance = 0.0) const;
    
    /**
     * Check if bootstrap phase is complete
     * @param trade_count Total trades executed
     * @return True if ready for dynamic leverage
     */
    bool is_bootstrap_complete(int trade_count) const;
    
private:
    static constexpr int PHASE_1_TRADES = 20;
    static constexpr int PHASE_2_TRADES = 50;
    static constexpr int PHASE_3_TRADES = 100;
    
    static constexpr double PHASE_1_LEVERAGE = 1.0;
    static constexpr double PHASE_2_LEVERAGE = 1.5;
    static constexpr double PHASE_3_LEVERAGE = 2.0;
};

} // namespace trading

#endif // SHARPE_TRACKER_H

#ifndef RISK_KERNELS_H
#define RISK_KERNELS_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Risk calculation functions
double calculate_kelly_position_size(double win_rate, double avg_win, double avg_loss, double account_balance);

double calculate_risk_adjusted_position(double signal_strength, double confidence, 
                                   double volatility, double account_balance, double risk_percent);

void calculate_portfolio_metrics(const double* returns, size_t len, const double* weights,
                            double* portfolio_return, double* portfolio_variance,
                            double* sharpe_ratio, double* max_drawdown);

// Advanced risk metrics
double calculate_value_at_risk(const double* returns, size_t len, double confidence_level);

double calculate_expected_shortfall(const double* returns, size_t len, double confidence_level);

void calculate_rolling_var(const double* returns, size_t len, size_t window,
                         double* rolling_var);

bool check_position_size_limits(double quantity, double price, double account_balance,
                           double max_position_pct, double min_notional);

#ifdef __cplusplus
}
#endif

#endif // RISK_KERNELS_H

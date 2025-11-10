#ifndef PORTFOLIO_OPT_H
#define PORTFOLIO_OPT_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Portfolio optimization functions
bool markowitz_optimization(const double* expected_returns, size_t n_assets,
                          const double* covariance_matrix, double target_return,
                          double min_weight, double max_weight, double* optimal_weights);

bool calculate_risk_parity_weights(const double* covariance_matrix, size_t n_assets,
                                 double* risk_parity_weights);

bool calculate_efficient_frontier(const double* expected_returns, size_t n_assets,
                                const double* covariance_matrix, size_t num_points,
                                double* frontier_returns, double* frontier_risks,
                                double* frontier_weights);

// Utility functions
double calculate_portfolio_return(const double* returns, const double* weights, size_t n_assets);

double calculate_portfolio_variance(const double* covariance_matrix, const double* weights, size_t n_assets);

bool optimize_portfolio_with_constraints(const double* expected_returns, size_t n_assets,
                                   const double* covariance_matrix,
                                   const double* min_weights, const double* max_weights,
                                   double* optimal_weights);

#ifdef __cplusplus
}
#endif

#endif // PORTFOLIO_OPT_H

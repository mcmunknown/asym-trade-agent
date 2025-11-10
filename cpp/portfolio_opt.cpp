#include "portfolio_opt.h"
#include <math.h>
#include <algorithm>
#include <vector>
#include <stdexcept>

namespace mathcore {

double calculate_portfolio_return(const double* returns, const double* weights, size_t n_assets) {
    double portfolio_return = 0.0;
    for (size_t i = 0; i < n_assets; ++i) {
        portfolio_return += returns[i] * weights[i];
    }
    return portfolio_return;
}

double calculate_portfolio_variance(const double* covariance_matrix, const double* weights, size_t n_assets) {
    double variance = 0.0;
    for (size_t i = 0; i < n_assets; ++i) {
        for (size_t j = 0; j < n_assets; ++j) {
            variance += weights[i] * covariance_matrix[i * n_assets + j] * weights[j];
        }
    }
    return variance;
}

bool markowitz_optimization(const double* expected_returns, size_t n_assets,
                          const double* covariance_matrix, double target_return,
                          double min_weight, double max_weight, double* optimal_weights) {
    if (!expected_returns || !covariance_matrix || !optimal_weights || n_assets == 0) {
        return false;
    }
    
    // Simplified Markowitz optimization using Lagrangian method
    // For full implementation, would use quadratic programming solver
    // This is a simplified version for demonstration
    
    // Initialize weights to equal weight
    double equal_weight = 1.0 / n_assets;
    for (size_t i = 0; i < n_assets; ++i) {
        optimal_weights[i] = equal_weight;
    }
    
    // Simple iterative improvement (gradient descent-like)
    const int max_iterations = 1000;
    const double learning_rate = 0.01;
    const double tolerance = 1e-6;
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Calculate gradient of objective function
        std::vector<double> gradient(n_assets, 0.0);
        
        // Gradient of portfolio variance
        for (size_t i = 0; i < n_assets; ++i) {
            for (size_t j = 0; j < n_assets; ++j) {
                gradient[i] += 2.0 * covariance_matrix[i * n_assets + j] * optimal_weights[j];
            }
        }
        
        // Adjust for return constraint using Lagrange multiplier
        double current_return = calculate_portfolio_return(expected_returns, optimal_weights, n_assets);
        double return_error = current_return - target_return;
        
        // Update weights with gradient descent
        double max_change = 0.0;
        for (size_t i = 0; i < n_assets; ++i) {
            double gradient_with_return = gradient[i] + return_error * expected_returns[i];
            double new_weight = optimal_weights[i] - learning_rate * gradient_with_return;
            
            // Apply constraints
            new_weight = std::max(min_weight, std::min(max_weight, new_weight));
            
            max_change = std::max(max_change, std::abs(new_weight - optimal_weights[i]));
            optimal_weights[i] = new_weight;
        }
        
        // Renormalize weights to sum to 1
        double sum_weights = 0.0;
        for (size_t i = 0; i < n_assets; ++i) {
            sum_weights += optimal_weights[i];
        }
        
        if (sum_weights > 0.0) {
            for (size_t i = 0; i < n_assets; ++i) {
                optimal_weights[i] /= sum_weights;
            }
        }
        
        if (max_change < tolerance) {
            break;
        }
    }
    
    return true;
}

bool calculate_risk_parity_weights(const double* covariance_matrix, size_t n_assets,
                                 double* risk_parity_weights) {
    if (!covariance_matrix || !risk_parity_weights || n_assets == 0) {
        return false;
    }
    
    // Risk parity: equal risk contribution from each asset
    // Risk contribution: RCᵢ = wᵢ·(Σw)ᵢ / √(wᵀΣw)
    
    // Initialize with equal weights
    double equal_weight = 1.0 / n_assets;
    for (size_t i = 0; i < n_assets; ++i) {
        risk_parity_weights[i] = equal_weight;
    }
    
    // Iterative solution to equalize risk contributions
    const int max_iterations = 1000;
    const double tolerance = 1e-6;
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Calculate current risk contributions
        double portfolio_variance = calculate_portfolio_variance(covariance_matrix, risk_parity_weights, n_assets);
        double portfolio_std = std::sqrt(portfolio_variance);
        
        std::vector<double> risk_contributions(n_assets, 0.0);
        for (size_t i = 0; i < n_assets; ++i) {
            double marginal_contrib = 0.0;
            for (size_t j = 0; j < n_assets; ++j) {
                marginal_contrib += covariance_matrix[i * n_assets + j] * risk_parity_weights[j];
            }
            risk_contributions[i] = risk_parity_weights[i] * marginal_contrib / portfolio_std;
        }
        
        // Update weights to equalize risk contributions
        double target_risk = 1.0 / n_assets; // Equal risk share
        
        double max_change = 0.0;
        for (size_t i = 0; i < n_assets; ++i) {
            if (risk_contributions[i] > 0.0) {
                double adjustment_factor = std::sqrt(target_risk / risk_contributions[i]);
                double new_weight = risk_parity_weights[i] * adjustment_factor;
                
                max_change = std::max(max_change, std::abs(new_weight - risk_parity_weights[i]));
                risk_parity_weights[i] = new_weight;
            }
        }
        
        // Renormalize weights
        double sum_weights = 0.0;
        for (size_t i = 0; i < n_assets; ++i) {
            sum_weights += risk_parity_weights[i];
        }
        
        if (sum_weights > 0.0) {
            for (size_t i = 0; i < n_assets; ++i) {
                risk_parity_weights[i] /= sum_weights;
            }
        }
        
        if (max_change < tolerance) {
            break;
        }
    }
    
    return true;
}

bool calculate_efficient_frontier(const double* expected_returns, size_t n_assets,
                                const double* covariance_matrix, size_t num_points,
                                double* frontier_returns, double* frontier_risks,
                                double* frontier_weights) {
    if (!expected_returns || !covariance_matrix || !frontier_returns || 
        !frontier_risks || !frontier_weights || n_assets == 0 || num_points == 0) {
        return false;
    }
    
    // Find min and max returns for target range
    double min_return = expected_returns[0];
    double max_return = expected_returns[0];
    for (size_t i = 1; i < n_assets; ++i) {
        min_return = std::min(min_return, expected_returns[i]);
        max_return = std::max(max_return, expected_returns[i]);
    }
    
    // Generate efficient frontier points
    for (size_t point = 0; point < num_points; ++point) {
        double target_return = min_return + (max_return - min_return) * point / (num_points - 1);
        
        // Store weights for this point
        double* weights = frontier_weights + point * n_assets;
        
        // Optimize for this target return
        if (markowitz_optimization(expected_returns, n_assets, covariance_matrix,
                                 target_return, 0.0, 1.0, weights)) {
            
            // Calculate corresponding risk
            double variance = calculate_portfolio_variance(covariance_matrix, weights, n_assets);
            
            frontier_returns[point] = target_return;
            frontier_risks[point] = std::sqrt(variance);
        } else {
            // Fallback if optimization fails
            frontier_returns[point] = 0.0;
            frontier_risks[point] = 0.0;
        }
    }
    
    return true;
}

bool optimize_portfolio_with_constraints(const double* expected_returns, size_t n_assets,
                                   const double* covariance_matrix,
                                   const double* min_weights, const double* max_weights,
                                   double* optimal_weights) {
    // This would implement constrained quadratic programming
    // For now, use simplified approach with basic constraints
    
    if (!expected_returns || !covariance_matrix || !optimal_weights || 
        !min_weights || !max_weights || n_assets == 0) {
        return false;
    }
    
    // Start with equal weights respecting constraints
    double sum_weights = 0.0;
    for (size_t i = 0; i < n_assets; ++i) {
        optimal_weights[i] = std::max(min_weights[i], std::min(max_weights[i], 1.0 / n_assets));
        sum_weights += optimal_weights[i];
    }
    
    // Normalize to sum to 1 while respecting constraints
    for (size_t iter = 0; iter < 100; ++iter) {
        bool adjusted = false;
        
        for (size_t i = 0; i < n_assets; ++i) {
            double target_weight = optimal_weights[i] / sum_weights;
            
            if (target_weight < min_weights[i]) {
                optimal_weights[i] = min_weights[i];
                adjusted = true;
            } else if (target_weight > max_weights[i]) {
                optimal_weights[i] = max_weights[i];
                adjusted = true;
            } else {
                optimal_weights[i] = target_weight;
            }
        }
        
        // Recalculate sum
        sum_weights = 0.0;
        for (size_t i = 0; i < n_assets; ++i) {
            sum_weights += optimal_weights[i];
        }
        
        if (!adjusted || std::abs(sum_weights - 1.0) < 1e-6) {
            break;
        }
    }
    
    return true;
}

} // namespace mathcore

// C interface for Python binding
extern "C" {

bool mc_markowitz_optimization(const double* expected_returns, size_t n_assets,
                            const double* covariance_matrix, double target_return,
                            double min_weight, double max_weight, double* optimal_weights) {
    return mathcore::markowitz_optimization(expected_returns, n_assets, covariance_matrix,
                                        target_return, min_weight, max_weight, optimal_weights);
}

bool mc_calculate_risk_parity_weights(const double* covariance_matrix, size_t n_assets,
                                    double* risk_parity_weights) {
    return mathcore::calculate_risk_parity_weights(covariance_matrix, n_assets, risk_parity_weights);
}

bool mc_calculate_efficient_frontier(const double* expected_returns, size_t n_assets,
                                   const double* covariance_matrix, size_t num_points,
                                   double* frontier_returns, double* frontier_risks,
                                   double* frontier_weights) {
    return mathcore::calculate_efficient_frontier(expected_returns, n_assets, covariance_matrix,
                                              num_points, frontier_returns, frontier_risks, frontier_weights);
}

double mc_calculate_portfolio_return(const double* returns, const double* weights, size_t n_assets) {
    return mathcore::calculate_portfolio_return(returns, weights, n_assets);
}

double mc_calculate_portfolio_variance(const double* covariance_matrix, const double* weights, size_t n_assets) {
    return mathcore::calculate_portfolio_variance(covariance_matrix, weights, n_assets);
}

bool mc_optimize_portfolio_with_constraints(const double* expected_returns, size_t n_assets,
                                        const double* covariance_matrix,
                                        const double* min_weights, const double* max_weights,
                                        double* optimal_weights) {
    return mathcore::optimize_portfolio_with_constraints(expected_returns, n_assets, covariance_matrix,
                                                 min_weights, max_weights, optimal_weights);
}

}

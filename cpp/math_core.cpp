#include "math_core.h"

// This file implements the C++ wrapper functions that call the C functions
// This allows us to maintain both C and C++ interfaces

namespace mathcore {

bool is_finite(double value) {
    return std::isfinite(value);
}

double safe_divide(double numerator, double denominator, double epsilon) {
    if (!std::isfinite(numerator) || !std::isfinite(denominator)) {
        return 0.0;
    }
    
    double abs_denominator = std::abs(denominator);
    if (abs_denominator < epsilon) {
        return 0.0;
    }
    
    return numerator / denominator;
}

double clamp_lambda(double lambda_param) {
    if (lambda_param <= 0.0) {
        return 1e-9;
    }
    if (lambda_param >= 1.0) {
        return 1.0 - 1e-9;
    }
    return lambda_param;
}

// C++ wrapper functions for NumPy integration
void exponential_smoothing(const double* prices, size_t len, double lambda_param, double* output) {
    mc_exponential_smoothing(prices, len, lambda_param, output);
}

void velocity(const double* smoothed, size_t len, double dt, double* output) {
    mc_velocity(smoothed, len, dt, output);
}

void acceleration(const double* velocity_series, size_t len, double dt, double* output) {
    mc_acceleration(velocity_series, len, dt, output);
}

void analyze_curve(const double* prices, size_t len, double lambda_param, double dt,
                  double* smoothed_out, double* velocity_out, double* acceleration_out) {
    mc_analyze_curve(prices, len, lambda_param, dt, smoothed_out, velocity_out, acceleration_out);
}

} // namespace mathcore

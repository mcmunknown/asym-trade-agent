#ifndef MATH_CORE_H
#define MATH_CORE_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Basic mathematical operations
void mc_exponential_smoothing(const double* prices,
                              size_t len,
                              double lambda_param,
                              double* output);

void mc_velocity(const double* smoothed,
                 size_t len,
                 double dt,
                 double* output);

void mc_acceleration(const double* velocity,
                     size_t len,
                     double dt,
                     double* output);

void mc_analyze_curve(const double* prices,
                      size_t len,
                      double lambda_param,
                      double dt,
                      double* smoothed_out,
                      double* velocity_out,
                      double* acceleration_out);

// Utility functions
bool mc_is_finite(double value);
double mc_safe_divide(double numerator, double denominator, double epsilon);
double mc_clamp_lambda(double lambda_param);

#ifdef __cplusplus
}
#endif

#endif // MATH_CORE_H

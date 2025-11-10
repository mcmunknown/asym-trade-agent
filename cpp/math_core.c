#include <stddef.h>
#include <math.h>
#include <fenv.h>

static double sanitize(double value, double fallback) {
    return isfinite(value) ? value : fallback;
}

static double clamp_lambda(double lambda_param) {
    if (lambda_param <= 0.0) {
        return 1e-9;
    }
    if (lambda_param >= 1.0) {
        return 1.0 - 1e-9;
    }
    return lambda_param;
}

static void reset_fp_env(void) {
#if defined(FE_DFL_ENV)
    fesetenv(FE_DFL_ENV);
#endif
}

void mc_exponential_smoothing(const double* prices,
                              size_t len,
                              double lambda_param,
                              double* output) {
    reset_fp_env();

    if (prices == NULL || output == NULL || len == 0) {
        return;
    }

    const double lambda = clamp_lambda(lambda_param);
    const double one_minus_lambda = 1.0 - lambda;

    double last_price = sanitize(prices[0], 0.0);
    output[0] = last_price;

    for (size_t i = 1; i < len; ++i) {
        const double price = sanitize(prices[i], last_price);
        const double smoothed = lambda * price + one_minus_lambda * output[i - 1];
        output[i] = smoothed;
        last_price = price;
    }
}

void mc_velocity(const double* smoothed,
                 size_t len,
                 double dt,
                 double* output) {
    reset_fp_env();

    if (smoothed == NULL || output == NULL || len == 0 || dt <= 0.0) {
        return;
    }

    output[0] = 0.0;
    const double inv_dt = 1.0 / dt;

    for (size_t i = 1; i < len; ++i) {
        const double current = sanitize(smoothed[i], smoothed[i - 1]);
        const double previous = sanitize(smoothed[i - 1], current);
        output[i] = (current - previous) * inv_dt;
    }
}

void mc_acceleration(const double* velocity,
                     size_t len,
                     double dt,
                     double* output) {
    reset_fp_env();

    if (velocity == NULL || output == NULL || len == 0 || dt <= 0.0) {
        return;
    }

    output[0] = 0.0;
    const double inv_dt = 1.0 / dt;

    for (size_t i = 1; i < len; ++i) {
        const double current = sanitize(velocity[i], velocity[i - 1]);
        const double previous = sanitize(velocity[i - 1], current);
        output[i] = (current - previous) * inv_dt;
    }
}

void mc_analyze_curve(const double* prices,
                      size_t len,
                      double lambda_param,
                      double dt,
                      double* smoothed_out,
                      double* velocity_out,
                      double* acceleration_out) {
    reset_fp_env();

    if (prices == NULL || smoothed_out == NULL || velocity_out == NULL ||
        acceleration_out == NULL || len == 0) {
        return;
    }

    mc_exponential_smoothing(prices, len, lambda_param, smoothed_out);
    mc_velocity(smoothed_out, len, dt, velocity_out);
    mc_acceleration(velocity_out, len, dt, acceleration_out);
}

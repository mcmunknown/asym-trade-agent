#include "math_core.h"
#include <algorithm>

// External C interface functions that call C++ implementations
extern "C" {

void mc_exponential_smoothing(const double* prices,
                              size_t len,
                              double lambda_param,
                              double* output) {
    if (!prices || !output || len == 0) return;
    
    double lambda = lambda_param;
    if (lambda <= 0.0) lambda = 1e-9;
    if (lambda >= 1.0) lambda = 1.0 - 1e-9;
    
    double one_minus_lambda = 1.0 - lambda;
    
    output[0] = prices[0];
    for (size_t i = 1; i < len; ++i) {
        output[i] = lambda * prices[i] + one_minus_lambda * output[i-1];
    }
}

void mc_velocity(const double* smoothed,
                 size_t len,
                 double dt,
                 double* output) {
    if (!smoothed || !output || len == 0 || dt <= 0.0) return;
    
    output[0] = 0.0;
    double inv_dt = 1.0 / dt;
    
    for (size_t i = 1; i < len; ++i) {
        output[i] = (smoothed[i] - smoothed[i-1]) * inv_dt;
    }
}

void mc_acceleration(const double* velocity,
                     size_t len,
                     double dt,
                     double* output) {
    if (!velocity || !output || len == 0 || dt <= 0.0) return;
    
    output[0] = 0.0;
    double inv_dt = 1.0 / dt;
    
    for (size_t i = 1; i < len; ++i) {
        output[i] = (velocity[i] - velocity[i-1]) * inv_dt;
    }
}

void mc_analyze_curve(const double* prices,
                      size_t len,
                      double lambda_param,
                      double dt,
                      double* smoothed_out,
                      double* velocity_out,
                      double* acceleration_out) {
    if (!prices || !smoothed_out || !velocity_out || !acceleration_out || len == 0) return;
    
    mc_exponential_smoothing(prices, len, lambda_param, smoothed_out);
    mc_velocity(smoothed_out, len, dt, velocity_out);
    mc_acceleration(velocity_out, len, dt, acceleration_out);
}

bool mc_is_finite(double value) {
    return std::isfinite(value);
}

double mc_safe_divide(double numerator, double denominator, double epsilon) {
    if (!std::isfinite(numerator) || !std::isfinite(denominator)) return 0.0;
    if (std::abs(denominator) < epsilon) return 0.0;
    return numerator / denominator;
}

double mc_clamp_lambda(double lambda_param) {
    if (lambda_param <= 0.0) return 1e-9;
    if (lambda_param >= 1.0) return 1.0 - 1e-9;
    return lambda_param;
}

} // extern "C"

// C++ wrapper functions for pybind11
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

double exponential_smoothing(py::array_t<double> prices, double lambda_param) {
    py::buffer_info buf = prices.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Number of dimensions must be one");
    }
    
    double* ptr = static_cast<double*>(buf.ptr);
    size_t len = buf.shape[0];
    
    if (len == 0) return 0.0;
    
    double result;
    mc_exponential_smoothing(ptr, len, lambda_param, &result);
    return result;
}

py::array_t<double> velocity_array(py::array_t<double> smoothed, double dt) {
    py::buffer_info buf = smoothed.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Number of dimensions must be one");
    }
    
    double* ptr = static_cast<double*>(buf.ptr);
    size_t len = buf.shape[0];
    
    auto result = py::array_t<double>(len);
    py::buffer_info result_buf = result.request();
    double* result_ptr = static_cast<double*>(result_buf.ptr);
    
    mc_velocity(ptr, len, dt, result_ptr);
    return result;
}

py::array_t<double> acceleration_array(py::array_t<double> velocity, double dt) {
    py::buffer_info buf = velocity.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Number of dimensions must be one");
    }
    
    double* ptr = static_cast<double*>(buf.ptr);
    size_t len = buf.shape[0];
    
    auto result = py::array_t<double>(len);
    py::buffer_info result_buf = result.request();
    double* result_ptr = static_cast<double*>(result_buf.ptr);
    
    mc_acceleration(ptr, len, dt, result_ptr);
    return result;
}

std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>> 
analyze_curve_complete(py::array_t<double> prices, double lambda_param, double dt) {
    py::buffer_info buf = prices.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Number of dimensions must be one");
    }
    
    double* ptr = static_cast<double*>(buf.ptr);
    size_t len = buf.shape[0];
    
    auto smoothed = py::array_t<double>(len);
    auto velocity = py::array_t<double>(len);
    auto acceleration = py::array_t<double>(len);
    
    py::buffer_info smooth_buf = smoothed.request();
    py::buffer_info vel_buf = velocity.request();
    py::buffer_info acc_buf = acceleration.request();
    
    mc_analyze_curve(ptr, len, lambda_param, dt,
                     static_cast<double*>(smooth_buf.ptr),
                     static_cast<double*>(vel_buf.ptr),
                     static_cast<double*>(acc_buf.ptr));
    
    return std::make_tuple(smoothed, velocity, acceleration);
}

bool cpp_available() {
    return true;
}

const char* version() {
    return "2.0.0-enhanced";
}

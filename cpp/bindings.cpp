#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "math_core.h"
#include "kalman_filter.h"
#include "risk_kernels.h"
#include "portfolio_opt.h"

namespace py = pybind11;

PYBIND11_MODULE(mathcore, m) {
    m.doc() = "High-performance mathematical kernels for calculus-based trading";

    // Basic math core functions
    m.def("exponential_smoothing", &exponential_smoothing,
           "Exponential smoothing: P̂ₜ = λ·Pₜ + (1-λ)·P̂ₜ₋₁",
           py::arg("prices"), py::arg("lambda_param"));
    
    m.def("velocity", &calculate_velocity,
           "First derivative: vₜ = (Pₜ - Pₜ₋₁)/Δt",
           py::arg("smoothed"), py::arg("dt"));
    
    m.def("acceleration", &calculate_acceleration,
           "Second derivative: aₜ = (vₜ - vₜ₋₁)/Δt", 
           py::arg("velocity"), py::arg("dt"));
    
    m.def("analyze_curve", &analyze_curve_complete,
           "Complete curve analysis: smoothing + velocity + acceleration",
           py::arg("prices"), py::arg("lambda_param"), py::arg("dt"));

    // Kalman filter functions
    py::class_<KalmanFilter>(m, "KalmanFilter")
        .def(py::init<double, double, double>())
        .def("update", &KalmanFilter::update)
        .def("get_state", &KalmanFilter::get_state)
        .def("reset", &KalmanFilter::reset);
    
    m.def("kalman_batch_filter", &kalman_batch_filter,
           "Batch Kalman filtering for price series",
           py::arg("prices"), py::arg("process_noise"), py::arg("observation_noise"), py::arg("dt"));

    // Risk calculation kernels
    m.def("kelly_position_size", &calculate_kelly_position_size,
           "Kelly criterion: f* = (p·b - q)/b",
           py::arg("win_rate"), py::arg("avg_win"), py::arg("avg_loss"), py::arg("account_balance"));
    
    m.def("risk_adjusted_position", &calculate_risk_adjusted_position,
           "Risk-adjusted position sizing",
           py::arg("signal_strength"), py::arg("confidence"), py::arg("volatility"), 
           py::arg("account_balance"), py::arg("risk_percent"));
    
    m.def("calculate_portfolio_risk", &calculate_portfolio_metrics,
           "Portfolio risk metrics: variance, sharpe, drawdown",
           py::arg("returns"), py::arg("weights"));

    // Portfolio optimization primitives
    m.def("markowitz_optimize", &markowitz_optimization,
           "Mean-variance optimization: min wᵀΣw subject to constraints",
           py::arg("expected_returns"), py::arg("covariance_matrix"), 
           py::arg("target_return"), py::arg("min_weight"), py::arg("max_weight"));
    
    m.def("risk_parity_weights", &calculate_risk_parity_weights,
           "Risk parity: equal risk contribution across assets",
           py::arg("covariance_matrix"));
    
    m.def("efficient_frontier", &calculate_efficient_frontier,
           "Calculate efficient frontier points",
           py::arg("expected_returns"), py::arg("covariance_matrix"), py::arg("num_points"));

    // Utility functions
    m.def("cpp_available", []() { return true; }, "Check if C++ backend is available");
    m.def("version", []() { return "2.0.0"; }, "C++ math core version");
}

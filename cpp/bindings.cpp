#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "math_core.h"
#include "kalman_filter.h"
#include "risk_kernels.h"
#include "portfolio_opt.h"
#include "ar_model.h"
#include "sharpe_tracker.h"
#include "order_book.h"
#include "market_maker.h"

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

    // AR(1) Linear Regression
    using namespace mathcore;
    
    py::class_<ARModelParams>(m, "ARModelParams")
        .def(py::init<>())
        .def_readwrite("weight", &ARModelParams::weight)
        .def_readwrite("bias", &ARModelParams::bias)
        .def_readwrite("r_squared", &ARModelParams::r_squared)
        .def_readwrite("regime_type", &ARModelParams::regime_type);
    
    py::class_<AR1LinearRegression>(m, "AR1LinearRegression")
        .def(py::init<size_t>(), py::arg("window_size") = 50)
        .def("fit_ols", [](AR1LinearRegression& self, py::array_t<double> log_returns) {
            auto buf = log_returns.request();
            return self.fit_ols(static_cast<double*>(buf.ptr), buf.size);
        }, "Fit AR(1) model using OLS: β = (X'X)^(-1) X'y",
           py::arg("log_returns"))
        .def("fit_gradient_descent", [](AR1LinearRegression& self, py::array_t<double> log_returns,
                                        double learning_rate, int n_iterations) {
            auto buf = log_returns.request();
            return self.fit_gradient_descent(static_cast<double*>(buf.ptr), buf.size,
                                            learning_rate, n_iterations);
        }, "Fit AR(1) model using gradient descent",
           py::arg("log_returns"), py::arg("learning_rate") = 0.01, py::arg("n_iterations") = 1000)
        .def("predict_next_return", &AR1LinearRegression::predict_next_return,
             "Predict next return: ŷ_{t+1} = w * y_t + b",
             py::arg("current_return"), py::arg("params"))
        .def("should_trade_mean_reversion", &AR1LinearRegression::should_trade_mean_reversion,
             "Check if mean reversion trade is justified",
             py::arg("params"), py::arg("min_r_squared") = 0.3)
        .def("should_trade_momentum", &AR1LinearRegression::should_trade_momentum,
             "Check if momentum trade is justified",
             py::arg("params"), py::arg("min_r_squared") = 0.3);
    
    // Strategy selection
    py::class_<StrategySelection>(m, "StrategySelection")
        .def(py::init<>())
        .def_readwrite("strategy_type", &StrategySelection::strategy_type)
        .def_readwrite("ar_params", &StrategySelection::ar_params)
        .def_readwrite("confidence", &StrategySelection::confidence);
    
    m.def("select_strategy", [](py::array_t<double> log_returns, int regime_state, double regime_confidence) {
        auto buf = log_returns.request();
        return select_strategy(static_cast<double*>(buf.ptr), buf.size, regime_state, regime_confidence);
    }, "Select trading strategy based on regime and AR(1) fit",
       py::arg("log_returns"), py::arg("regime_state"), py::arg("regime_confidence"));
    
    // =============================================
    // SHARPE TRACKER (Phase 4)
    // =============================================
    py::class_<trading::SharpeTracker>(m, "SharpeTracker")
        .def(py::init<int, double, int>(),
             "Real-time Sharpe ratio tracker",
             py::arg("window_size") = 100,
             py::arg("risk_free_rate") = 0.04,
             py::arg("periods_per_year") = 365)
        .def("add_return", &trading::SharpeTracker::add_return,
             "Add a trade return",
             py::arg("trade_return"))
        .def("calculate_sharpe", &trading::SharpeTracker::calculate_sharpe,
             "Calculate current Sharpe ratio")
        .def("get_recommended_leverage", &trading::SharpeTracker::get_recommended_leverage,
             "Get Sharpe-based leverage recommendation",
             py::arg("max_leverage") = 10.0)
        .def("get_mean_return", &trading::SharpeTracker::get_mean_return,
             "Get mean return")
        .def("get_volatility", &trading::SharpeTracker::get_volatility,
             "Get return volatility")
        .def("get_trade_count", &trading::SharpeTracker::get_trade_count,
             "Get number of trades tracked")
        .def("get_win_rate", &trading::SharpeTracker::get_win_rate,
             "Get win rate")
        .def("has_sufficient_data", &trading::SharpeTracker::has_sufficient_data,
             "Check if sufficient data for reliable calculation")
        .def("reset", &trading::SharpeTracker::reset,
             "Reset tracker");
    
    py::class_<trading::LeverageBootstrap>(m, "LeverageBootstrap")
        .def(py::init<>(),
             "Conservative leverage ramp-up for new systems")
        .def("get_bootstrap_leverage", &trading::LeverageBootstrap::get_bootstrap_leverage,
             "Get bootstrap leverage for trade count and balance",
             py::arg("trade_count"), py::arg("account_balance") = 0.0)
        .def("is_bootstrap_complete", &trading::LeverageBootstrap::is_bootstrap_complete,
             "Check if bootstrap phase complete",
             py::arg("trade_count"));
    
    // =============================================
    // ORDER BOOK (Phase 5)
    // =============================================
    py::class_<trading::OrderBook>(m, "OrderBook")
        .def(py::init<>(),
             "Order book parser and analyzer")
        .def("update", [](trading::OrderBook& ob, py::array_t<double> bids, py::array_t<double> asks) {
            auto bids_buf = bids.request();
            auto asks_buf = asks.request();
            ob.update(static_cast<double*>(bids_buf.ptr), bids_buf.size,
                     static_cast<double*>(asks_buf.ptr), asks_buf.size);
        }, "Update order book with bids/asks",
           py::arg("bids"), py::arg("asks"))
        .def("get_mid_price", &trading::OrderBook::get_mid_price,
             "Get mid-price (best bid + best ask) / 2")
        .def("get_weighted_mid_price", &trading::OrderBook::get_weighted_mid_price,
             "Get volume-weighted mid-price")
        .def("get_spread", &trading::OrderBook::get_spread,
             "Get spread (best ask - best bid)")
        .def("get_spread_bps", &trading::OrderBook::get_spread_bps,
             "Get spread in basis points")
        .def("get_imbalance", &trading::OrderBook::get_imbalance,
             "Get order book imbalance (-1 to +1)")
        .def("get_bid_volume", &trading::OrderBook::get_bid_volume,
             "Get total bid volume",
             py::arg("levels") = 5)
        .def("get_ask_volume", &trading::OrderBook::get_ask_volume,
             "Get total ask volume",
             py::arg("levels") = 5)
        .def("get_depth_ratio", &trading::OrderBook::get_depth_ratio,
             "Get market depth ratio")
        .def("is_valid", &trading::OrderBook::is_valid,
             "Check if order book is valid")
        .def("get_best_bid", &trading::OrderBook::get_best_bid,
             "Get best bid price")
        .def("get_best_ask", &trading::OrderBook::get_best_ask,
             "Get best ask price");
    
    py::class_<trading::OrderBookSignal>(m, "OrderBookSignal")
        .def_static("calculate_signal", &trading::OrderBookSignal::calculate_signal,
                   "Calculate microstructure signal",
                   py::arg("imbalance"), py::arg("spread_bps"), py::arg("depth_ratio"))
        .def_static("is_spread_acceptable", &trading::OrderBookSignal::is_spread_acceptable,
                   "Check if spread is acceptable",
                   py::arg("spread_bps"), py::arg("max_spread_bps") = 20.0)
        .def_static("calculate_slippage", &trading::OrderBookSignal::calculate_slippage,
                   "Calculate expected slippage",
                   py::arg("order_book"), py::arg("side"), py::arg("notional"));
    
    // =============================================
    // MARKET MAKER (Phase 6)
    // =============================================
    py::class_<trading::MarketMaker::Quote>(m, "MarketMakerQuote")
        .def(py::init<>())
        .def_readwrite("bid_price", &trading::MarketMaker::Quote::bid_price)
        .def_readwrite("bid_size", &trading::MarketMaker::Quote::bid_size)
        .def_readwrite("ask_price", &trading::MarketMaker::Quote::ask_price)
        .def_readwrite("ask_size", &trading::MarketMaker::Quote::ask_size)
        .def_readwrite("spread", &trading::MarketMaker::Quote::spread)
        .def_readwrite("edge", &trading::MarketMaker::Quote::edge);
    
    py::class_<trading::MarketMaker>(m, "MarketMaker")
        .def_static("calculate_quotes", &trading::MarketMaker::calculate_quotes,
                   "Calculate optimal market maker quotes",
                   py::arg("mid_price"),
                   py::arg("volatility"),
                   py::arg("prediction"),
                   py::arg("inventory"),
                   py::arg("spread_multiplier") = 2.0,
                   py::arg("max_position_size") = 1.0)
        .def_static("calculate_inventory_skew", &trading::MarketMaker::calculate_inventory_skew,
                   "Calculate inventory skew",
                   py::arg("inventory"), py::arg("max_skew_bps") = 10.0)
        .def_static("calculate_prediction_bias", &trading::MarketMaker::calculate_prediction_bias,
                   "Calculate prediction bias",
                   py::arg("prediction"), py::arg("max_bias_bps") = 5.0)
        .def_static("calculate_quote_size", &trading::MarketMaker::calculate_quote_size,
                   "Calculate quote size",
                   py::arg("confidence"), py::arg("inventory"), py::arg("max_size"))
        .def_static("should_flatten_inventory", &trading::MarketMaker::should_flatten_inventory,
                   "Check if inventory needs flattening",
                   py::arg("inventory"), py::arg("threshold") = 0.8);
    
    py::class_<trading::InventoryManager>(m, "InventoryManager")
        .def(py::init<double>(),
             "Inventory manager for market making",
             py::arg("max_position_size"))
        .def("update_position", &trading::InventoryManager::update_position,
             "Update current position",
             py::arg("position"))
        .def("get_normalized_inventory", &trading::InventoryManager::get_normalized_inventory,
             "Get normalized inventory (-1 to +1)")
        .def("get_position", &trading::InventoryManager::get_position,
             "Get current position")
        .def("get_max_position", &trading::InventoryManager::get_max_position,
             "Get maximum position size")
        .def("can_go_longer", &trading::InventoryManager::can_go_longer,
             "Check if can add to long position",
             py::arg("size"))
        .def("can_go_shorter", &trading::InventoryManager::can_go_shorter,
             "Check if can add to short position",
             py::arg("size"))
        .def("get_flatten_urgency", &trading::InventoryManager::get_flatten_urgency,
             "Get urgency to flatten inventory (0-1)");

    // Utility functions
    m.def("cpp_available", []() { return true; }, "Check if C++ backend is available");
    m.def("version", []() { return "2.0.0"; }, "C++ math core version");
}

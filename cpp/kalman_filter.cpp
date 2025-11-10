#include "kalman_filter.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace mathcore {

KalmanFilter::KalmanFilter(double process_noise_price,
                         double process_noise_velocity,
                         double process_noise_acceleration,
                         double observation_noise,
                         double dt)
    : observation_noise_(observation_noise), dt_(dt), initialized_(false) {
    
    // Initialize state vector [price, velocity, acceleration]
    state_.resize(3, 0.0);
    
    // Initialize covariance matrix (3x3)
    covariance_.resize(9, 0.0);
    covariance_[0] = 1.0; // Initial price uncertainty
    covariance_[4] = 1.0; // Initial velocity uncertainty  
    covariance_[8] = 1.0; // Initial acceleration uncertainty
    
    // Initialize process noise matrix (diagonal)
    process_noise_.resize(9, 0.0);
    process_noise_[0] = process_noise_price;
    process_noise_[4] = process_noise_velocity;
    process_noise_[8] = process_noise_acceleration;
    
    // Initialize temporary matrices
    temp_matrix_.resize(9, 0.0);
    kalman_gain_.resize(3, 0.0);
    
    initialize_matrices();
}

void KalmanFilter::initialize_matrices() {
    // State transition matrix A for constant acceleration model:
    // [1, dt, 0.5*dt²]
    // [0, 1,  dt     ]
    // [0, 0,   1      ]
    transition_matrix_.resize(9, 0.0);
    transition_matrix_[0] = 1.0;
    transition_matrix_[1] = dt_;
    transition_matrix_[2] = 0.5 * dt_ * dt_;
    transition_matrix_[4] = 1.0;
    transition_matrix_[5] = dt_;
    transition_matrix_[8] = 1.0;
    
    // Observation matrix H = [1, 0, 0]
    observation_matrix_.resize(3, 0.0);
    observation_matrix_[0] = 1.0;
}

void KalmanFilter::update(double price_observation) {
    if (!initialized_) {
        state_[0] = price_observation; // Initial price
        initialized_ = true;
        return;
    }
    
    predict();
    update_step(price_observation);
}

void KalmanFilter::predict() {
    // Predict step: sₜ|ₜ₋₁ = A·sₜ₋₁|ₜ₋₁
    std::vector<double> predicted_state(3, 0.0);
    
    // Matrix multiplication: predicted_state = A * state_
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            predicted_state[i] += transition_matrix_[i * 3 + j] * state_[j];
        }
    }
    
    // Predict covariance: P̂ₜ|ₜ₋₁ = A·P̂ₜ₋₁|ₜ₋₁·Aᵀ + Q
    // temp_matrix_ = A * covariance_
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            temp_matrix_[i * 3 + j] = 0.0;
            for (int k = 0; k < 3; ++k) {
                temp_matrix_[i * 3 + j] += transition_matrix_[i * 3 + k] * covariance_[k * 3 + j];
            }
        }
    }
    
    // covariance_ = temp_matrix_ * A^T + Q
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            covariance_[i * 3 + j] = 0.0;
            for (int k = 0; k < 3; ++k) {
                covariance_[i * 3 + j] += temp_matrix_[i * 3 + k] * transition_matrix_[j * 3 + k];
            }
            covariance_[i * 3 + j] += process_noise_[i * 3 + j];
        }
    }
    
    state_ = predicted_state;
}

void KalmanFilter::update_step(double observation) {
    // Calculate Kalman gain: K = P·Hᵀ·(H·P·Hᵀ + R)⁻¹
    
    // H·P·Hᵀ = covariance_[0,0] (since H = [1,0,0])
    double innovation_covariance = covariance_[0] + observation_noise_;
    
    // Kalman gain: K = P·Hᵀ / innovation_covariance
    for (int i = 0; i < 3; ++i) {
        kalman_gain_[i] = covariance_[i * 3] / innovation_covariance;
    }
    
    // Innovation: observation - H·state = observation - price
    double innovation = observation - state_[0];
    
    // Update state: sₜ|ₜ = sₜ|ₜ₋₁ + K·innovation
    for (int i = 0; i < 3; ++i) {
        state_[i] += kalman_gain_[i] * innovation;
    }
    
    // Update covariance: P = (I - K·H)·P
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double kh = (i == 0) ? kalman_gain_[0] : 0.0;
            covariance_[i * 3 + j] = (i == j ? 1.0 : 0.0) - kh;
        }
    }
    
    // temp_matrix_ = (I - K·H) * covariance_
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            temp_matrix_[i * 3 + j] = 0.0;
            for (int k = 0; k < 3; ++k) {
                double kh = (k == 0) ? kalman_gain_[i] : 0.0;
                temp_matrix_[i * 3 + j] += ((i == k) ? 1.0 : 0.0) - kh) * covariance_[k * 3 + j];
            }
        }
    }
    
    covariance_ = temp_matrix_;
}

void KalmanFilter::batch_filter(const double* prices, size_t len,
                             double* filtered_prices, double* velocities, double* accelerations) {
    reset(prices[0]);
    
    for (size_t i = 0; i < len; ++i) {
        update(prices[i]);
        
        // Store results
        get_state(&filtered_prices[i], &velocities[i], &accelerations[i]);
    }
}

void KalmanFilter::get_state(double* price, double* velocity, double* acceleration) const {
    if (price) *price = state_[0];
    if (velocity) *velocity = state_[1];
    if (acceleration) *acceleration = state_[2];
}

void KalmanFilter::get_uncertainty(double* price_uncertainty, double* velocity_uncertainty,
                                double* acceleration_uncertainty) const {
    if (price_uncertainty) *price_uncertainty = std::sqrt(covariance_[0]);
    if (velocity_uncertainty) *velocity_uncertainty = std::sqrt(covariance_[4]);
    if (acceleration_uncertainty) *acceleration_uncertainty = std::sqrt(covariance_[8]);
}

void KalmanFilter::reset(double initial_price) {
    state_[0] = initial_price;
    state_[1] = 0.0; // velocity
    state_[2] = 0.0; // acceleration
    
    // Reset covariance to initial values
    covariance_[0] = 1.0; // price uncertainty
    covariance_[4] = 1.0; // velocity uncertainty
    covariance_[8] = 1.0; // acceleration uncertainty
    
    initialized_ = true;
}

// Batch processing function for Python integration
void kalman_batch_filter(const double* prices, size_t len,
                       double process_noise, double observation_noise, double dt,
                       double* filtered_prices, double* velocities, double* accelerations) {
    KalmanFilter filter(process_noise, process_noise * 0.1, process_noise * 0.01, 
                      observation_noise, dt);
    filter.batch_filter(prices, len, filtered_prices, velocities, accelerations);
}

} // namespace mathcore

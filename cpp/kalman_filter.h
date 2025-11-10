#ifndef KALMAN_FILTER_H
#define KALMAN_FILTER_H

#include <vector>

namespace mathcore {

class KalmanFilter {
public:
    KalmanFilter(double process_noise_price = 1e-5,
                double process_noise_velocity = 1e-6,
                double process_noise_acceleration = 1e-7,
                double observation_noise = 1e-4,
                double dt = 1.0);
    
    // Single-step update
    void update(double price_observation);
    
    // Batch processing for price series
    void batch_filter(const double* prices, size_t len, 
                    double* filtered_prices, double* velocities, double* accelerations);
    
    // Get current state estimates
    void get_state(double* price, double* velocity, double* acceleration) const;
    
    // Get uncertainty estimates
    void get_uncertainty(double* price_uncertainty, double* velocity_uncertainty, 
                      double* acceleration_uncertainty) const;
    
    // Reset filter to initial state
    void reset(double initial_price = 0.0);
    
    // Check if filter is initialized
    bool is_initialized() const { return initialized_; }

private:
    // State vector: [price, velocity, acceleration]
    std::vector<double> state_;
    
    // Covariance matrix (3x3 stored as flat array)
    std::vector<double> covariance_;
    
    // Process noise covariance matrix (3x3)
    std::vector<double> process_noise_;
    
    // Observation noise scalar
    double observation_noise_;
    
    // State transition matrix (3x3)
    std::vector<double> transition_matrix_;
    
    // Observation matrix [1, 0, 0]
    std::vector<double> observation_matrix_;
    
    // Time step
    double dt_;
    
    // Temporary matrices for calculations
    std::vector<double> temp_matrix_;
    std::vector<double> kalman_gain_;
    
    bool initialized_;
    
    void initialize_matrices();
    void predict();
    void update_step(double observation);
};

// Batch processing function for Python integration
void kalman_batch_filter(const double* prices, size_t len,
                       double process_noise, double observation_noise, double dt,
                       double* filtered_prices, double* velocities, double* accelerations);

} // namespace mathcore

#endif // KALMAN_FILTER_H

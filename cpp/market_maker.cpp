#include "market_maker.h"

namespace trading {

// ============================================
// MarketMaker Implementation
// ============================================

MarketMaker::Quote MarketMaker::calculate_quotes(double mid_price,
                                                  double volatility,
                                                  double prediction,
                                                  double inventory,
                                                  double spread_multiplier,
                                                  double max_position_size) {
    Quote quote;
    
    if (mid_price <= 0 || volatility <= 0) {
        return quote;  // Invalid inputs
    }
    
    // Base spread (multiple of volatility)
    double base_spread = volatility * spread_multiplier;
    double half_spread = base_spread / 2.0;
    
    // Calculate inventory skew (wider on inventory side)
    double inv_skew = calculate_inventory_skew(inventory, 10.0);
    
    // Calculate prediction bias (shift in prediction direction)
    double pred_bias = calculate_prediction_bias(prediction, 5.0);
    
    // Apply skew and bias to half-spread
    double bid_offset = half_spread + inv_skew - pred_bias;
    double ask_offset = half_spread - inv_skew + pred_bias;
    
    // Calculate quote prices
    quote.bid_price = mid_price - bid_offset;
    quote.ask_price = mid_price + ask_offset;
    quote.spread = quote.ask_price - quote.bid_price;
    
    // Calculate sizes based on confidence and inventory
    double confidence = std::abs(prediction);
    quote.bid_size = calculate_quote_size(confidence, inventory, max_position_size);
    quote.ask_size = calculate_quote_size(confidence, -inventory, max_position_size);
    
    // Expected edge (spread capture)
    quote.edge = quote.spread / mid_price;  // As fraction
    
    return quote;
}

double MarketMaker::calculate_inventory_skew(double inventory, double max_skew_bps) {
    // Clamp inventory to [-1, 1]
    inventory = std::max(-1.0, std::min(1.0, inventory));
    
    // Linear skew: positive inventory → positive skew (widen bids)
    double skew_fraction = inventory * (max_skew_bps / 10000.0);
    
    return skew_fraction;
}

double MarketMaker::calculate_prediction_bias(double prediction, double max_bias_bps) {
    // Clamp prediction to [-1, 1]
    prediction = std::max(-1.0, std::min(1.0, prediction));
    
    // Bias in direction of prediction
    double bias_fraction = prediction * (max_bias_bps / 10000.0);
    
    return bias_fraction;
}

double MarketMaker::calculate_quote_size(double confidence,
                                         double inventory,
                                         double max_size) {
    // Base size from confidence
    double base_size = confidence * max_size;
    
    // Reduce size when inventory is large in same direction
    // If inventory > 0 and we're calculating bid size (adding long), reduce
    // If inventory < 0 and we're calculating ask size (adding short), reduce
    double inv_factor = 1.0 - std::abs(inventory);
    
    return base_size * inv_factor;
}

bool MarketMaker::should_flatten_inventory(double inventory, double threshold) {
    return std::abs(inventory) > threshold;
}

// ============================================
// InventoryManager Implementation
// ============================================

InventoryManager::InventoryManager(double max_position_size)
    : position_(0.0),
      max_position_size_(max_position_size) {
}

void InventoryManager::update_position(double position) {
    position_ = position;
}

double InventoryManager::get_normalized_inventory() const {
    if (max_position_size_ <= 0) {
        return 0.0;
    }
    
    double normalized = position_ / max_position_size_;
    return std::max(-1.0, std::min(1.0, normalized));
}

double InventoryManager::get_position() const {
    return position_;
}

double InventoryManager::get_max_position() const {
    return max_position_size_;
}

bool InventoryManager::can_go_longer(double size) const {
    return (position_ + size) <= max_position_size_;
}

bool InventoryManager::can_go_shorter(double size) const {
    return (position_ - size) >= -max_position_size_;
}

double InventoryManager::get_flatten_urgency() const {
    double abs_inventory = std::abs(get_normalized_inventory());
    
    if (abs_inventory < 0.5) {
        return 0.0;  // No urgency
    } else if (abs_inventory < 0.8) {
        return (abs_inventory - 0.5) / 0.3;  // Linear ramp
    } else {
        return 1.0;  // Max urgency
    }
}

} // namespace trading

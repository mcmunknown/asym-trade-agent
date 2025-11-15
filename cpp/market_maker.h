#ifndef MARKET_MAKER_H
#define MARKET_MAKER_H

#include <cmath>
#include <algorithm>

namespace trading {

/**
 * Market Maker Strategy
 * 
 * Provides liquidity by placing bid/ask quotes around mid-price.
 * Captures spread while managing inventory risk.
 * 
 * Key concepts:
 * - Spread: Bid-ask difference (profit per round-trip)
 * - Bias: Shift quotes based on prediction (directional edge)
 * - Inventory: Position size affects quote placement
 * - Skew: Asymmetric quotes to manage inventory
 */
class MarketMaker {
public:
    /**
     * Quote structure (bid and ask prices with sizes)
     */
    struct Quote {
        double bid_price;
        double bid_size;
        double ask_price;
        double ask_size;
        double spread;
        double edge;  // Expected profit per round-trip
        
        Quote() : bid_price(0), bid_size(0), ask_price(0), ask_size(0), spread(0), edge(0) {}
    };
    
    /**
     * Calculate optimal market maker quotes
     * 
     * @param mid_price Current mid-price
     * @param volatility Recent volatility (e.g., ATR)
     * @param prediction AR(1) or model prediction (-1 to +1)
     * @param inventory Current inventory (-1 = max short, +1 = max long, 0 = neutral)
     * @param spread_multiplier Base spread as multiple of volatility (default: 2.0)
     * @param max_position_size Maximum position size
     * @return Quote with bid/ask prices and sizes
     */
    static Quote calculate_quotes(double mid_price,
                                  double volatility,
                                  double prediction,
                                  double inventory,
                                  double spread_multiplier = 2.0,
                                  double max_position_size = 1.0);
    
    /**
     * Calculate inventory skew
     * 
     * Skew quotes to encourage inventory reduction:
     * - Long inventory → widen bids, tighten asks (encourage selling)
     * - Short inventory → tighten bids, widen asks (encourage buying)
     * 
     * @param inventory Normalized inventory (-1 to +1)
     * @param max_skew_bps Maximum skew in basis points (default: 10)
     * @return Skew factor for bid/ask adjustment
     */
    static double calculate_inventory_skew(double inventory, double max_skew_bps = 10.0);
    
    /**
     * Calculate prediction bias
     * 
     * Shift quotes in direction of prediction:
     * - Bullish prediction → wider bids, tighter asks
     * - Bearish prediction → tighter bids, wider asks
     * 
     * @param prediction Model prediction (-1 to +1)
     * @param max_bias_bps Maximum bias in basis points (default: 5)
     * @return Bias factor for quote adjustment
     */
    static double calculate_prediction_bias(double prediction, double max_bias_bps = 5.0);
    
    /**
     * Calculate position size based on confidence and inventory
     * 
     * @param confidence Signal confidence (0 to 1)
     * @param inventory Current inventory (-1 to +1)
     * @param max_size Maximum position size
     * @return Recommended quote size
     */
    static double calculate_quote_size(double confidence,
                                       double inventory,
                                       double max_size);
    
    /**
     * Check if inventory is too large (needs flattening)
     * 
     * @param inventory Normalized inventory (-1 to +1)
     * @param threshold Inventory threshold (default: 0.8 = 80%)
     * @return True if inventory should be reduced
     */
    static bool should_flatten_inventory(double inventory, double threshold = 0.8);
};

/**
 * Inventory Manager
 * 
 * Tracks and manages market maker inventory risk
 */
class InventoryManager {
public:
    InventoryManager(double max_position_size);
    
    /**
     * Update current position
     * @param position Current position size (positive = long, negative = short)
     */
    void update_position(double position);
    
    /**
     * Get normalized inventory (-1 to +1)
     */
    double get_normalized_inventory() const;
    
    /**
     * Get current position
     */
    double get_position() const;
    
    /**
     * Get maximum position size
     */
    double get_max_position() const;
    
    /**
     * Check if can add to long position
     * @param size Size to add
     * @return True if within limits
     */
    bool can_go_longer(double size) const;
    
    /**
     * Check if can add to short position
     * @param size Size to add
     * @return True if within limits
     */
    bool can_go_shorter(double size) const;
    
    /**
     * Calculate urgency to flatten inventory (0 to 1)
     * Higher = more urgent to reduce position
     */
    double get_flatten_urgency() const;
    
private:
    double position_;
    double max_position_size_;
};

} // namespace trading

#endif // MARKET_MAKER_H

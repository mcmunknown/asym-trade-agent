#ifndef ORDER_BOOK_H
#define ORDER_BOOK_H

#include <vector>
#include <algorithm>
#include <cmath>

namespace trading {

/**
 * Order Book Level (single price level)
 */
struct OrderBookLevel {
    double price;
    double quantity;
    
    OrderBookLevel(double p, double q) : price(p), quantity(q) {}
};

/**
 * Order Book Parser and Analyzer
 * 
 * Efficiently parses order book data and calculates market microstructure metrics:
 * - Mid-price (best bid + best ask) / 2
 * - Weighted mid-price (volume-weighted)
 * - Spread (best ask - best bid)
 * - Imbalance (bid volume - ask volume) / total volume
 * - Market depth (liquidity at various price levels)
 * 
 * Performance target: < 2μs per update
 */
class OrderBook {
public:
    OrderBook();
    
    /**
     * Update order book with new bids/asks
     * @param bids Array of [price, quantity] pairs (sorted descending)
     * @param asks Array of [price, quantity] pairs (sorted ascending)
     * @param bid_count Number of bid levels
     * @param ask_count Number of ask levels
     */
    void update(const double* bids, int bid_count,
                const double* asks, int ask_count);
    
    /**
     * Get current mid-price (simple average of best bid/ask)
     */
    double get_mid_price() const;
    
    /**
     * Get volume-weighted mid-price (weighted by top 5 levels)
     */
    double get_weighted_mid_price() const;
    
    /**
     * Get current spread (best ask - best bid)
     */
    double get_spread() const;
    
    /**
     * Get spread in basis points (spread / mid_price * 10000)
     */
    double get_spread_bps() const;
    
    /**
     * Get order book imbalance (-1 to +1)
     * Positive = more bids (bullish), Negative = more asks (bearish)
     */
    double get_imbalance() const;
    
    /**
     * Get total bid volume (top N levels)
     */
    double get_bid_volume(int levels = 5) const;
    
    /**
     * Get total ask volume (top N levels)
     */
    double get_ask_volume(int levels = 5) const;
    
    /**
     * Get market depth ratio (volume at 1% price distance)
     */
    double get_depth_ratio() const;
    
    /**
     * Check if order book is valid (has both bids and asks)
     */
    bool is_valid() const;
    
    /**
     * Get best bid price
     */
    double get_best_bid() const;
    
    /**
     * Get best ask price
     */
    double get_best_ask() const;
    
private:
    std::vector<OrderBookLevel> bids_;
    std::vector<OrderBookLevel> asks_;
    
    // Cached values
    mutable double cached_mid_price_;
    mutable double cached_imbalance_;
    mutable bool cache_valid_;
    
    void invalidate_cache();
    void update_cache() const;
};

/**
 * Order Book Signal Generator
 * 
 * Generates trading signals from order book microstructure:
 * - Large imbalance → expect price movement in imbalance direction
 * - Widening spread → reduced liquidity, avoid trading
 * - Depth changes → detect large orders
 */
class OrderBookSignal {
public:
    /**
     * Calculate microstructure signal strength (-1 to +1)
     * @param imbalance Order book imbalance
     * @param spread_bps Spread in basis points
     * @param depth_ratio Market depth ratio
     * @return Signal strength (positive = bullish, negative = bearish)
     */
    static double calculate_signal(double imbalance, double spread_bps, double depth_ratio);
    
    /**
     * Check if spread is too wide for safe trading
     * @param spread_bps Spread in basis points
     * @param max_spread_bps Maximum acceptable spread
     * @return True if spread is acceptable
     */
    static bool is_spread_acceptable(double spread_bps, double max_spread_bps = 20.0);
    
    /**
     * Calculate expected slippage for market order
     * @param order_book Current order book
     * @param side Buy or sell (1 = buy, -1 = sell)
     * @param notional Order notional value
     * @return Expected slippage in price units
     */
    static double calculate_slippage(const OrderBook& order_book, int side, double notional);
};

} // namespace trading

#endif // ORDER_BOOK_H

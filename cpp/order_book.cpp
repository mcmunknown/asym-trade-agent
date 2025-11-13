#include "order_book.h"
#include <numeric>
#include <cmath>
#include <algorithm>

namespace trading {

// ============================================
// OrderBook Implementation
// ============================================

OrderBook::OrderBook()
    : cached_mid_price_(0.0),
      cached_imbalance_(0.0),
      cache_valid_(false) {
}

void OrderBook::update(const double* bids, int bid_count,
                       const double* asks, int ask_count) {
    bids_.clear();
    asks_.clear();
    
    // Parse bids (price, quantity pairs)
    for (int i = 0; i < bid_count; i += 2) {
        if (i + 1 < bid_count) {
            bids_.emplace_back(bids[i], bids[i + 1]);
        }
    }
    
    // Parse asks (price, quantity pairs)
    for (int i = 0; i < ask_count; i += 2) {
        if (i + 1 < ask_count) {
            asks_.emplace_back(asks[i], asks[i + 1]);
        }
    }
    
    invalidate_cache();
}

double OrderBook::get_mid_price() const {
    if (!is_valid()) {
        return 0.0;
    }
    
    double best_bid = bids_[0].price;
    double best_ask = asks_[0].price;
    
    return (best_bid + best_ask) / 2.0;
}

double OrderBook::get_weighted_mid_price() const {
    if (!is_valid()) {
        return 0.0;
    }
    
    // Volume-weighted average of top 5 levels
    int levels = std::min(5, static_cast<int>(std::min(bids_.size(), asks_.size())));
    
    double bid_weighted_sum = 0.0;
    double bid_volume = 0.0;
    for (int i = 0; i < levels && i < static_cast<int>(bids_.size()); i++) {
        bid_weighted_sum += bids_[i].price * bids_[i].quantity;
        bid_volume += bids_[i].quantity;
    }
    
    double ask_weighted_sum = 0.0;
    double ask_volume = 0.0;
    for (int i = 0; i < levels && i < static_cast<int>(asks_.size()); i++) {
        ask_weighted_sum += asks_[i].price * asks_[i].quantity;
        ask_volume += asks_[i].quantity;
    }
    
    if (bid_volume + ask_volume < 1e-10) {
        return get_mid_price();
    }
    
    double weighted_bid = bid_weighted_sum / bid_volume;
    double weighted_ask = ask_weighted_sum / ask_volume;
    
    return (weighted_bid + weighted_ask) / 2.0;
}

double OrderBook::get_spread() const {
    if (!is_valid()) {
        return 0.0;
    }
    
    return asks_[0].price - bids_[0].price;
}

double OrderBook::get_spread_bps() const {
    double spread = get_spread();
    double mid = get_mid_price();
    
    if (mid < 1e-10) {
        return 0.0;
    }
    
    return (spread / mid) * 10000.0;  // Basis points
}

double OrderBook::get_imbalance() const {
    update_cache();
    return cached_imbalance_;
}

double OrderBook::get_bid_volume(int levels) const {
    double total = 0.0;
    int count = std::min(levels, static_cast<int>(bids_.size()));
    
    for (int i = 0; i < count; i++) {
        total += bids_[i].quantity;
    }
    
    return total;
}

double OrderBook::get_ask_volume(int levels) const {
    double total = 0.0;
    int count = std::min(levels, static_cast<int>(asks_.size()));
    
    for (int i = 0; i < count; i++) {
        total += asks_[i].quantity;
    }
    
    return total;
}

double OrderBook::get_depth_ratio() const {
    if (!is_valid()) {
        return 0.0;
    }
    
    double mid = get_mid_price();
    double price_threshold = mid * 0.01;  // 1% price distance
    
    // Calculate volume within 1% of mid price
    double bid_depth = 0.0;
    for (const auto& level : bids_) {
        if (mid - level.price <= price_threshold) {
            bid_depth += level.quantity;
        } else {
            break;
        }
    }
    
    double ask_depth = 0.0;
    for (const auto& level : asks_) {
        if (level.price - mid <= price_threshold) {
            ask_depth += level.quantity;
        } else {
            break;
        }
    }
    
    double total_depth = bid_depth + ask_depth;
    double total_volume = get_bid_volume(10) + get_ask_volume(10);
    
    if (total_volume < 1e-10) {
        return 0.0;
    }
    
    return total_depth / total_volume;
}

bool OrderBook::is_valid() const {
    return !bids_.empty() && !asks_.empty();
}

double OrderBook::get_best_bid() const {
    return is_valid() ? bids_[0].price : 0.0;
}

double OrderBook::get_best_ask() const {
    return is_valid() ? asks_[0].price : 0.0;
}

void OrderBook::invalidate_cache() {
    cache_valid_ = false;
}

void OrderBook::update_cache() const {
    if (cache_valid_ || !is_valid()) {
        return;
    }
    
    // Calculate mid-price
    cached_mid_price_ = get_mid_price();
    
    // Calculate imbalance
    double bid_vol = get_bid_volume(5);
    double ask_vol = get_ask_volume(5);
    double total_vol = bid_vol + ask_vol;
    
    if (total_vol < 1e-10) {
        cached_imbalance_ = 0.0;
    } else {
        cached_imbalance_ = (bid_vol - ask_vol) / total_vol;
    }
    
    cache_valid_ = true;
}

// ============================================
// OrderBookSignal Implementation
// ============================================

double OrderBookSignal::calculate_signal(double imbalance, double spread_bps, double depth_ratio) {
    // Signal strength based on imbalance
    double signal = imbalance;
    
    // Penalize wide spreads (low liquidity)
    if (spread_bps > 10.0) {
        double spread_penalty = 1.0 - std::min(1.0, (spread_bps - 10.0) / 20.0);
        signal *= spread_penalty;
    }
    
    // Boost signal with good depth
    signal *= (0.5 + 0.5 * depth_ratio);
    
    // Clamp to [-1, 1]
    return std::max(-1.0, std::min(1.0, signal));
}

bool OrderBookSignal::is_spread_acceptable(double spread_bps, double max_spread_bps) {
    return spread_bps <= max_spread_bps && spread_bps > 0.0;
}

double OrderBookSignal::calculate_slippage(const OrderBook& order_book, int side, double notional) {
    if (!order_book.is_valid()) {
        return 0.0;
    }
    
    double mid = order_book.get_mid_price();
    double remaining_notional = notional;
    double weighted_price = 0.0;
    double filled_quantity = 0.0;
    
    if (side > 0) {
        // Buy side - walk the ask book
        for (int i = 0; i < 10 && remaining_notional > 0; i++) {
            double level_notional = order_book.get_best_ask() * 100;  // Simplified
            double fill = std::min(remaining_notional, level_notional);
            weighted_price += order_book.get_best_ask() * fill;
            filled_quantity += fill;
            remaining_notional -= fill;
        }
    } else {
        // Sell side - walk the bid book
        for (int i = 0; i < 10 && remaining_notional > 0; i++) {
            double level_notional = order_book.get_best_bid() * 100;  // Simplified
            double fill = std::min(remaining_notional, level_notional);
            weighted_price += order_book.get_best_bid() * fill;
            filled_quantity += fill;
            remaining_notional -= fill;
        }
    }
    
    if (filled_quantity < 1e-10) {
        return 0.0;
    }
    
    double avg_fill_price = weighted_price / filled_quantity;
    return std::abs(avg_fill_price - mid);
}

} // namespace trading

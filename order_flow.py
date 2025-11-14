"""
Order Flow Imbalance Analysis
==============================

Renaissance-style structural edge from order book dynamics.

Mathematical Foundation:
    X_t = log(BuyVolume / AskLiquidity)
    E[r_{t+Δ}] = β₀ + β₁ X_t
    
    Where β₁ < 0 for mean reversion (buy pressure → pullback)
          β₁ > 0 for momentum (buy pressure → continuation)
    
    Order book imbalance: when buying pressure >> ask liquidity,
    price likely to rise sharply, then mean-revert. This is the edge.
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple, List
from collections import deque

logger = logging.getLogger(__name__)

class OrderFlowAnalyzer:
    """
    Analyzes order flow imbalance to detect buying/selling pressure.
    
    Uses volume delta (aggressive buys vs aggressive sells) as proxy
    for institutional order flow.
    """
    
    def __init__(self, window_size: int = 50):
        """
        Initialize order flow analyzer.
        
        Args:
            window_size: Rolling window for imbalance calculation
        """
        self.window_size = window_size
        self.buy_volumes = {}  # symbol -> deque of buy volumes
        self.sell_volumes = {}  # symbol -> deque of sell volumes
        self.timestamps = {}  # symbol -> deque of timestamps
        
    def update(self, symbol: str, trades: list):
        """
        Update order flow with latest trades.
        
        Args:
            symbol: Trading symbol
            trades: List of trade dicts with 'side', 'size', 'timestamp'
        """
        if symbol not in self.buy_volumes:
            self.buy_volumes[symbol] = deque(maxlen=self.window_size)
            self.sell_volumes[symbol] = deque(maxlen=self.window_size)
            self.timestamps[symbol] = deque(maxlen=self.window_size)
        
        for trade in trades:
            side = trade.get('side', '').lower()
            size = float(trade.get('size', 0))
            timestamp = trade.get('timestamp', 0)
            
            if side == 'buy':
                self.buy_volumes[symbol].append(size)
                self.sell_volumes[symbol].append(0)
            elif side == 'sell':
                self.buy_volumes[symbol].append(0)
                self.sell_volumes[symbol].append(size)
            else:
                # Unknown side, skip
                continue
                
            self.timestamps[symbol].append(timestamp)
    
    def calculate_imbalance(self, symbol: str) -> Optional[float]:
        """
        Calculate order flow imbalance.
        
        Formula: X_t = log(BuyVolume / SellVolume)
        
        Returns:
            Imbalance score:
                > 0: Buying pressure (bullish)
                < 0: Selling pressure (bearish)
                None: Insufficient data
        """
        if symbol not in self.buy_volumes:
            return None
        
        buy_vols = list(self.buy_volumes[symbol])
        sell_vols = list(self.sell_volumes[symbol])
        
        if len(buy_vols) < 10:  # Need minimum data
            return None
        
        total_buy = sum(buy_vols)
        total_sell = sum(sell_vols)
        
        if total_buy == 0 and total_sell == 0:
            return 0.0
        
        if total_sell == 0:
            return 1.0  # Max bullish
        
        if total_buy == 0:
            return -1.0  # Max bearish
        
        # Log imbalance (Renaissance formula)
        imbalance = np.log(total_buy / total_sell)
        
        # Normalize to [-1, 1] range
        # Typical imbalance is -2 to +2, so divide by 2
        normalized = np.tanh(imbalance / 2.0)
        
        return normalized
    
    def get_volume_delta(self, symbol: str) -> Optional[float]:
        """
        Calculate raw volume delta.
        
        Returns:
            Volume delta: buy_volume - sell_volume
        """
        if symbol not in self.buy_volumes:
            return None
        
        buy_vols = list(self.buy_volumes[symbol])
        sell_vols = list(self.sell_volumes[symbol])
        
        if len(buy_vols) < 10:
            return None
        
        return sum(buy_vols) - sum(sell_vols)
    
    def should_confirm_long(self, symbol: str, threshold: float = 0.1) -> bool:
        """
        Check if order flow confirms a LONG entry.
        
        Args:
            symbol: Trading symbol
            threshold: Minimum imbalance for confirmation (default 0.1)
        
        Returns:
            True if buying pressure confirms long entry
        """
        imbalance = self.calculate_imbalance(symbol)
        
        if imbalance is None:
            return True  # No data = don't block trade
        
        # For mean reversion: Want SELLING pressure when entering LONG
        # (Price fell due to selling, expect bounce)
        # For this, we want imbalance < -threshold
        
        # But for momentum: Want BUYING pressure when entering LONG
        # (Price rising with buyers, expect continuation)
        # For this, we want imbalance > threshold
        
        # Let's use a hybrid: Slight selling OK (mean reversion setup)
        return imbalance > -0.3  # Not excessive selling
    
    def should_confirm_short(self, symbol: str, threshold: float = 0.1) -> bool:
        """
        Check if order flow confirms a SHORT entry.
        
        Args:
            symbol: Trading symbol
            threshold: Minimum imbalance for confirmation
        
        Returns:
            True if selling pressure confirms short entry
        """
        imbalance = self.calculate_imbalance(symbol)
        
        if imbalance is None:
            return True  # No data = don't block trade
        
        # For mean reversion SHORT: Want BUYING pressure
        # (Price rose due to buying, expect pullback)
        return imbalance < 0.3  # Not excessive buying
    
    def get_signal_quality_adjustment(self, symbol: str, position_side: str) -> float:
        """
        Adjust signal quality based on order flow.
        
        Returns:
            Multiplier for confidence: 0.8 (weak) to 1.2 (strong)
        """
        imbalance = self.calculate_imbalance(symbol)
        
        if imbalance is None:
            return 1.0  # Neutral if no data
        
        # Check if order flow aligns with position
        if position_side.lower() == 'long' or position_side.lower() == 'buy':
            # For long: Moderate selling pressure is good (mean reversion setup)
            # Strong buying is also good (momentum setup)
            if -0.3 < imbalance < -0.1:
                return 1.1  # Good mean reversion setup
            elif imbalance > 0.2:
                return 1.15  # Strong momentum setup
            elif imbalance < -0.5:
                return 0.9  # Too much selling pressure
            else:
                return 1.0  # Neutral
        
        elif position_side.lower() == 'short' or position_side.lower() == 'sell':
            # For short: Moderate buying pressure is good (mean reversion setup)
            if 0.1 < imbalance < 0.3:
                return 1.1  # Good mean reversion setup
            elif imbalance < -0.2:
                return 1.15  # Strong selling momentum
            elif imbalance > 0.5:
                return 0.9  # Too much buying pressure
            else:
                return 1.0  # Neutral
        
        return 1.0
    
    def get_stats(self, symbol: str) -> Dict:
        """Get current order flow statistics for symbol."""
        imbalance = self.calculate_imbalance(symbol)
        delta = self.get_volume_delta(symbol)
        
        if symbol not in self.buy_volumes:
            return {
                'imbalance': None,
                'volume_delta': None,
                'buy_volume': 0,
                'sell_volume': 0,
                'sample_count': 0
            }
        
        return {
            'imbalance': imbalance,
            'volume_delta': delta,
            'buy_volume': sum(self.buy_volumes[symbol]),
            'sell_volume': sum(self.sell_volumes[symbol]),
            'sample_count': len(self.buy_volumes[symbol])
        }


class OrderBookImbalanceAnalyzer:
    """
    Renaissance-style order book imbalance detector.
    
    Tracks bid/ask imbalance from order book snapshots.
    Formula: X_t = log(buy_volume / ask_liquidity)
    
    When X_t is large (buying pressure >> ask liquidity):
        Price rises sharply → mean reversion likely → SHORT edge
    When X_t is small/negative (selling pressure >> bid liquidity):
        Price falls sharply → mean reversion likely → LONG edge
    """
    
    def __init__(self, window_size: int = 60):
        """
        Initialize order book analyzer.
        
        Args:
            window_size: Rolling window of order book snapshots
        """
        self.window_size = window_size
        
        # Track order book history per symbol
        self.bid_volumes = {}  # symbol -> deque of bid volumes
        self.ask_volumes = {}  # symbol -> deque of ask volumes
        self.bid_prices = {}   # symbol -> deque of bid prices
        self.ask_prices = {}   # symbol -> deque of ask prices
        self.timestamps = {}   # symbol -> deque of timestamps
        
        # Linear regression coefficients (learned from history)
        self.imbalance_betas = {}  # symbol -> (beta0, beta1) where r = beta0 + beta1*X
        self.imbalance_history = {}  # symbol -> deque of (timestamp, imbalance, return)
        
    def update_orderbook(self, symbol: str, bids: List[Tuple[float, float]], 
                        asks: List[Tuple[float, float]], timestamp: float):
        """
        Update order book snapshot.
        
        Args:
            symbol: Trading symbol
            bids: List of (price, volume) tuples sorted by price (highest first)
            asks: List of (price, volume) tuples sorted by price (lowest first)
            timestamp: Current timestamp
        """
        if symbol not in self.bid_volumes:
            self.bid_volumes[symbol] = deque(maxlen=self.window_size)
            self.ask_volumes[symbol] = deque(maxlen=self.window_size)
            self.bid_prices[symbol] = deque(maxlen=self.window_size)
            self.ask_prices[symbol] = deque(maxlen=self.window_size)
            self.timestamps[symbol] = deque(maxlen=self.window_size)
            self.imbalance_history[symbol] = deque(maxlen=self.window_size)
        
        # Calculate volume at each level
        total_bid_volume = sum(vol for _, vol in bids[:5]) if bids else 0  # Top 5 bid levels
        total_ask_volume = sum(vol for _, vol in asks[:5]) if asks else 0  # Top 5 ask levels
        
        mid_price = (bids[0][0] + asks[0][0]) / 2 if bids and asks else 0
        
        self.bid_volumes[symbol].append(total_bid_volume)
        self.ask_volumes[symbol].append(total_ask_volume)
        self.bid_prices[symbol].append(bids[0][0] if bids else 0)
        self.ask_prices[symbol].append(asks[0][0] if asks else 0)
        self.timestamps[symbol].append(timestamp)
    
    def calculate_book_imbalance(self, symbol: str) -> Optional[float]:
        """
        Calculate order book imbalance: log(bid_vol / ask_vol)
        
        Returns:
            Imbalance score:
                > 0: More bid liquidity (buyers stronger)
                < 0: More ask liquidity (sellers stronger)
                None: Insufficient data
        """
        if symbol not in self.bid_volumes or len(self.bid_volumes[symbol]) < 5:
            return None
        
        total_bid = sum(self.bid_volumes[symbol])
        total_ask = sum(self.ask_volumes[symbol])
        
        if total_bid == 0 or total_ask == 0:
            return 0.0
        
        # Log imbalance (Renaissance formula)
        imbalance = np.log(total_bid / total_ask)
        
        # Normalize to [-1, 1] using tanh
        normalized = np.tanh(imbalance / 2.0)
        
        return normalized
    
    def get_imbalance_signal(self, symbol: str, threshold: float = 0.2) -> Optional[str]:
        """
        Generate signal from order book imbalance.
        
        Args:
            symbol: Trading symbol
            threshold: Threshold for strong imbalance signal
        
        Returns:
            'LONG' if strong seller imbalance (mean reversion opportunity)
            'SHORT' if strong buyer imbalance (mean reversion opportunity)
            None if imbalance weak or no data
        """
        imbalance = self.calculate_book_imbalance(symbol)
        
        if imbalance is None:
            return None
        
        # Strong selling pressure (more asks than bids) → price fell → LONG edge
        if imbalance < -threshold:
            return 'LONG'
        
        # Strong buying pressure (more bids than asks) → price rose → SHORT edge
        elif imbalance > threshold:
            return 'SHORT'
        
        return None
    
    def should_gate_entry(self, symbol: str, direction: str, 
                         min_imbalance: float = 0.05) -> bool:
        """
        Gate trade entry based on order book imbalance alignment.
        
        Args:
            symbol: Trading symbol
            direction: 'LONG' or 'SHORT'
            min_imbalance: Minimum imbalance magnitude to allow trade
        
        Returns:
            True if order book supports entry, False otherwise
        """
        imbalance = self.calculate_book_imbalance(symbol)
        
        if imbalance is None:
            return True  # Allow if no data
        
        # For LONG: Want seller imbalance (imbalance < 0)
        if direction.upper() == 'LONG':
            return imbalance < min_imbalance  # Allow negative or weak positive
        
        # For SHORT: Want buyer imbalance (imbalance > 0)
        elif direction.upper() == 'SHORT':
            return imbalance > -min_imbalance  # Allow positive or weak negative
        
        return True
    
    def get_entry_confidence_boost(self, symbol: str, direction: str) -> float:
        """
        Get confidence multiplier from order book alignment.
        
        Returns:
            Multiplier 0.8 (weak) to 1.3 (strong)
        """
        imbalance = self.calculate_book_imbalance(symbol)
        
        if imbalance is None:
            return 1.0
        
        if direction.upper() == 'LONG':
            # Want negative imbalance (sellers stronger)
            if imbalance < -0.3:
                return 1.2  # Strong alignment
            elif imbalance < -0.1:
                return 1.1  # Moderate alignment
            elif imbalance < 0.1:
                return 1.0  # Neutral
            else:
                return 0.9  # Weak alignment
        
        elif direction.upper() == 'SHORT':
            # Want positive imbalance (buyers stronger)
            if imbalance > 0.3:
                return 1.2  # Strong alignment
            elif imbalance > 0.1:
                return 1.1  # Moderate alignment
            elif imbalance > -0.1:
                return 1.0  # Neutral
            else:
                return 0.9  # Weak alignment
        
        return 1.0
    
    def get_stats(self, symbol: str) -> Dict:
        """Get current order book imbalance statistics."""
        imbalance = self.calculate_book_imbalance(symbol)
        signal = self.get_imbalance_signal(symbol)
        
        if symbol not in self.bid_volumes:
            return {
                'book_imbalance': None,
                'signal': None,
                'bid_volume': 0,
                'ask_volume': 0,
                'sample_count': 0
            }
        
        return {
            'book_imbalance': imbalance,
            'signal': signal,
            'bid_volume': sum(self.bid_volumes[symbol]),
            'ask_volume': sum(self.ask_volumes[symbol]),
            'sample_count': len(self.bid_volumes[symbol])
        }

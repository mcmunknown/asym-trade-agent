"""
Order Flow Imbalance Analysis
==============================

Renaissance-style structural edge from order book dynamics.

Mathematical Foundation:
    X_t = log(BuyVolume / SellVolume)
    E[r_{t+Δ}] = β₀ + β₁ X_t
    
    Where β₁ < 0 for mean reversion (buy pressure → pullback)
          β₁ > 0 for momentum (buy pressure → continuation)
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple
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

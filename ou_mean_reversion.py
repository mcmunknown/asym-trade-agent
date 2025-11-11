"""
Ornstein-Uhlenbeck Mean Reversion Model
========================================

Renaissance-style mathematical timing for mean reversion entries.

Mathematical Foundation:
    dr_t = θ(μ - r_t)dt + σdW_t
    
    Where:
        θ = mean reversion speed (how fast price returns to mean)
        μ = long-term mean
        σ = volatility
        
    Half-life: t_½ = ln(2) / θ
    
    Optimal entry: Within 1 half-life of deviation
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)

class OUMeanReversionModel:
    """
    Ornstein-Uhlenbeck process for mean reversion timing.
    
    Calculates when price is likely to revert based on:
    1. Mean reversion speed (θ)
    2. Half-life of deviations
    3. Current distance from mean
    """
    
    def __init__(self, lookback: int = 100):
        """
        Initialize OU model.
        
        Args:
            lookback: Number of periods for parameter estimation
        """
        self.lookback = lookback
        self.price_history = {}  # symbol -> deque of prices
        self.params_cache = {}  # symbol -> (theta, mu, sigma, half_life)
        self.last_update = {}  # symbol -> timestamp
        
    def update_prices(self, symbol: str, prices: np.ndarray):
        """
        Update price history for symbol.
        
        Args:
            symbol: Trading symbol
            prices: Array of recent prices
        """
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.lookback)
        
        # Add new prices
        for price in prices:
            self.price_history[symbol].append(price)
    
    def estimate_ou_parameters(self, symbol: str) -> Optional[Tuple[float, float, float, float]]:
        """
        Estimate OU process parameters from price data.
        
        Uses discrete approximation:
            Δr_t = -θ(r_t - μ)Δt + σ√Δt ε_t
        
        Returns:
            (theta, mu, sigma, half_life) or None if insufficient data
        """
        if symbol not in self.price_history:
            return None
        
        prices = np.array(list(self.price_history[symbol]))
        
        if len(prices) < 30:  # Need minimum data
            return None
        
        # Calculate log returns
        log_prices = np.log(prices)
        returns = np.diff(log_prices)
        
        if len(returns) < 20:
            return None
        
        # Estimate parameters using regression
        # Model: Δr_t = α + β*r_t + ε
        # Where θ = -β, μ = -α/β
        
        lagged_returns = returns[:-1]
        delta_returns = np.diff(returns)
        
        if len(lagged_returns) == 0 or len(delta_returns) == 0:
            return None
        
        # Linear regression: Δr ~ r_lagged
        try:
            # Add constant term
            X = np.column_stack([np.ones(len(lagged_returns)), lagged_returns])
            y = delta_returns
            
            # Solve: β = (X'X)^-1 X'y
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            
            alpha = beta[0]
            beta_coef = beta[1]
            
            # Extract OU parameters
            theta = -beta_coef  # Mean reversion speed
            
            if theta <= 0:
                theta = 0.01  # Minimum positive theta
            
            mu = -alpha / theta if theta > 0 else 0  # Long-term mean
            
            # Estimate volatility
            residuals = y - (alpha + beta_coef * lagged_returns)
            sigma = np.std(residuals)
            
            # Calculate half-life
            half_life = np.log(2) / theta if theta > 0 else np.inf
            
            # Cache results
            self.params_cache[symbol] = (theta, mu, sigma, half_life)
            
            return (theta, mu, sigma, half_life)
            
        except Exception as e:
            logger.error(f"Error estimating OU parameters for {symbol}: {e}")
            return None
    
    def get_mean_reversion_signal(self, symbol: str, current_price: float) -> Dict:
        """
        Generate mean reversion signal with OU timing.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
        
        Returns:
            Dict with:
                - should_trade: bool (within mean reversion window)
                - deviation: float (distance from mean in std devs)
                - time_to_revert: float (expected seconds to mean)
                - confidence: float (0-1, higher = stronger signal)
        """
        params = self.estimate_ou_parameters(symbol)
        
        if params is None:
            return {
                'should_trade': True,  # Don't block if no data
                'deviation': 0,
                'time_to_revert': 0,
                'confidence': 0.5,
                'half_life': None
            }
        
        theta, mu, sigma, half_life = params
        
        # Calculate current deviation from mean
        if symbol not in self.price_history or len(self.price_history[symbol]) == 0:
            return {'should_trade': True, 'deviation': 0, 'time_to_revert': 0, 'confidence': 0.5, 'half_life': half_life}
        
        prices = np.array(list(self.price_history[symbol]))
        log_current = np.log(current_price)
        log_mean = np.mean(np.log(prices))
        
        deviation = (log_current - log_mean) / sigma if sigma > 0 else 0
        
        # Estimate time to revert (exponential decay)
        # E[r_t] = μ + (r_0 - μ)e^(-θt)
        # Solve for t when r_t = μ (mean)
        if abs(deviation) < 0.1:
            time_to_revert = 0  # Already at mean
        else:
            # Time to revert to 50% of deviation
            time_to_revert = half_life if half_life < np.inf else 0
        
        # Calculate confidence based on deviation and half-life
        # Stronger signal when:
        # 1. Larger deviation (but not too large)
        # 2. Shorter half-life (faster reversion expected)
        
        # Optimal deviation: 1-2 standard deviations
        deviation_score = 0
        abs_dev = abs(deviation)
        if 0.5 < abs_dev < 3.0:
            # Good range for mean reversion
            deviation_score = min(abs_dev / 2.0, 1.0)
        elif abs_dev >= 3.0:
            # Too extreme, might be regime change
            deviation_score = 0.5
        else:
            # Too close to mean, small profit potential
            deviation_score = abs_dev
        
        # Half-life score: Prefer 5-30 minutes
        if half_life < np.inf:
            if 300 < half_life < 1800:  # 5-30 min
                half_life_score = 1.0
            elif half_life < 300:  # Too fast, noisy
                half_life_score = 0.7
            else:  # Too slow
                half_life_score = 0.5
        else:
            half_life_score = 0.3
        
        # Combined confidence
        confidence = (deviation_score + half_life_score) / 2.0
        
        # Should trade if:
        # 1. Deviation is meaningful (> 0.5 std)
        # 2. Half-life is reasonable (< 1 hour)
        should_trade = abs_dev > 0.5 and half_life < 3600
        
        return {
            'should_trade': should_trade,
            'deviation': deviation,
            'time_to_revert': time_to_revert,
            'confidence': confidence,
            'half_life': half_life,
            'theta': theta,
            'mean': log_mean
        }
    
    def get_optimal_entry_side(self, symbol: str, current_price: float) -> Optional[str]:
        """
        Determine optimal entry side based on OU model.
        
        Returns:
            'long' if price below mean (expect reversion up)
            'short' if price above mean (expect reversion down)
            None if no clear signal
        """
        signal = self.get_mean_reversion_signal(symbol, current_price)
        
        if not signal['should_trade']:
            return None
        
        deviation = signal['deviation']
        
        if deviation < -0.5:
            # Price below mean → expect bounce
            return 'long'
        elif deviation > 0.5:
            # Price above mean → expect pullback
            return 'short'
        else:
            return None
    
    def get_stats(self, symbol: str, current_price: float = None) -> Dict:
        """Get current OU model statistics."""
        if current_price is None and symbol in self.price_history and len(self.price_history[symbol]) > 0:
            current_price = list(self.price_history[symbol])[-1]
        
        if current_price is None:
            return {
                'theta': None,
                'half_life': None,
                'deviation': None,
                'should_trade': False
            }
        
        signal = self.get_mean_reversion_signal(symbol, current_price)
        params = self.params_cache.get(symbol, (None, None, None, None))
        
        return {
            'theta': params[0],
            'mu': params[1],
            'sigma': params[2],
            'half_life': params[3],
            'deviation': signal['deviation'],
            'confidence': signal['confidence'],
            'should_trade': signal['should_trade'],
            'time_to_revert': signal.get('time_to_revert', 0)
        }

#!/usr/bin/env python3

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)

class CalculusTradingStrategy:
    """
    Anne's Fixed Calculus-Based Trading Strategy
    
    6️⃣ Decision Matrix:
    - v>0, a>0: trend accelerating → STRONG_BUY
    - v>0, a<0: momentum fading → TAKE_PROFIT  
    - v<0, a<0: downtrend accelerating → STRONG_SELL
    - v<0, a>0: bottom forming → LOOK_FOR_REVERSAL
    """
    
    def __init__(self, min_snr: float = 0.5, min_confidence: float = 50.0, lambda_param: float = 0.75):
        self.min_snr = min_snr
        self.min_confidence = min_confidence
        self.lambda_param = 0.75  # EMA smoothing
        
    def get_latest_signal(self, symbol: str, prices: pd.Series) -> Dict:
        """
        Get calculus-based trading signal with PROPER data alignment.
        
        FIXES: "Length of values (101) does not match length of index (100)"
        """
        try:
            # Ensure proper pandas Series with aligned index
            if not isinstance(prices, pd.Series):
                prices = pd.Series(prices)
            
            # Remove NaN values and ensure proper alignment
            clean_prices = prices.dropna()
            
            # Need minimum data for derivatives
            if len(clean_prices) < 3:
                return {
                    'signal': 0,
                    'confidence': 0.0,
                    'velocity': 0.0,
                    'acceleration': 0.0,
                    'snr': 0.0,
                    'ema_price': float(prices.iloc[-1]) if len(prices) > 0 else None
                }
            
            # EMA smoothing (creates proper aligned index)
            ema_prices = clean_prices.ewm(alpha=self.lambda_param).mean()
            ema_clean = ema_prices.dropna()
            
            # First derivative (velocity) with PROPER alignment
            velocity = ema_clean.diff().dropna()
            
            # Second derivative (acceleration) with PROPER alignment
            acceleration = velocity.diff().dropna()
            
            # Get latest aligned values
            if len(velocity) > 0 and len(acceleration) > 0:
                latest_velocity = float(velocity.iloc[-1])
                latest_acceleration = float(acceleration.iloc[-1])
                
                # Signal significance (SNR)
                if len(velocity) > 1:
                    noise_std = float(velocity.std())
                    snr = abs(latest_velocity) / noise_std if noise_std > 0 else 0.0
                else:
                    snr = 0.0
                
                # 6-Case Decision Matrix
                signal = 0  # Default: no signal
                
                if latest_velocity > 0 and latest_acceleration > 0:
                    signal = 1  # STRONG_BUY: trend accelerating
                elif latest_velocity > 0 and latest_acceleration < 0:
                    signal = 2  # TAKE_PROFIT: momentum fading
                elif latest_velocity < 0 and latest_acceleration < 0:
                    signal = -1  # STRONG_SELL: downtrend accelerating
                elif latest_velocity < 0 and latest_acceleration > 0:
                    signal = 0  # POSSIBLE_EXIT_SHORT: bottom forming
                elif abs(latest_velocity) < 0.01:  # Near zero velocity
                    signal = 3 if latest_acceleration > 0 else 0  # LOOK_FOR_REVERSAL
                
                # Confidence based on SNR and signal strength
                confidence = min(abs(snr) * 0.5 + abs(latest_velocity) * 10, 1.0)
                
                return {
                    'signal': signal,
                    'confidence': confidence,
                    'velocity': latest_velocity,
                    'acceleration': latest_acceleration,
                    'snr': snr,
                    'ema_price': float(ema_clean.iloc[-1]) if len(ema_clean) > 0 else None
                }
            
            return {
                'signal': 0,
                'confidence': 0.0,
                'velocity': 0.0,
                'acceleration': 0.0,
                'snr': 0.0,
                'ema_price': float(clean_prices.iloc[-1]) if len(clean_prices) > 0 else None
            }
            
        except Exception as e:
            logger.error(f"Calculus signal error for {symbol}: {e}")
            return {
                'signal': 0,
                'confidence': 0.0,
                'velocity': 0.0,
                'acceleration': 0.0,
                'snr': 0.0,
                'ema_price': None
            }
    
    def analyze_curve_geometry(self, symbol: str, prices: pd.Series) -> Dict:
        """Analyze geometric properties of price curve."""
        signal = self.get_latest_signal(symbol, prices)
        
        return {
            'symbol': symbol,
            'geometry': {
                'slope': signal.get('velocity', 0),
                'curvature': signal.get('acceleration', 0),
                'snr': signal.get('snr', 0)
            },
            'trading': signal
        }
    
    def detect_crossovers(self, prices: pd.Series) -> List[Dict]:
        """Detect crossovers between price and EMA."""
        try:
            ema = prices.ewm(alpha=self.lambda_param).mean()
            clean_ema = ema.dropna()
            clean_prices = prices.dropna()
            
            # Align series
            min_length = min(len(clean_prices), len(clean_ema))
            if min_length < 2:
                return []
                
            aligned_prices = clean_prices.iloc[-min_length:]
            aligned_ema = clean_ema.iloc[-min_length:]
            
            # Find crossovers
            crossovers = []
            for i in range(1, min_length):
                prev_diff = aligned_prices.iloc[i-1] - aligned_ema.iloc[i-1]
                curr_diff = aligned_prices.iloc[i] - aligned_ema.iloc[i]
                
                if prev_diff * curr_diff < 0:  # Sign change = crossover
                    crossovers.append({
                        'index': i,
                        'type': 'bullish' if curr_diff > 0 else 'bearish',
                        'price': float(aligned_prices.iloc[i]),
                        'ema': float(aligned_ema.iloc[i])
                    })
            
            return crossovers
        except Exception as e:
            logger.error(f"Crossover detection error: {e}")
            return []
    
    def generate_trading_signals(self, symbols: List[str], price_data: Dict[str, pd.Series]) -> Dict:
        """Generate trading signals for multiple symbols."""
        signals = {}
        
        for symbol in symbols:
            if symbol in price_data:
                prices = price_data[symbol]
                signal = self.get_latest_signal(symbol, prices)
                
                # Apply thresholds
                if (signal['confidence'] >= self.min_confidence / 100.0 and 
                    signal['snr'] >= self.min_snr and
                    signal['signal'] != 0):
                    signals[symbol] = signal
        
        return signals

# Add missing SignalType enum for imports
from enum import Enum

class SignalType(Enum):
    STRONG_BUY = 1
    TAKE_PROFIT = 2
    STRONG_SELL = -1
    LOOK_FOR_REVERSAL = 3
    HOLD = 0
    TRAIL_STOP_UP = 4
    HOLD_SHORT = 5
    POSSIBLE_LONG = 6
    POSSIBLE_EXIT_SHORT = 7

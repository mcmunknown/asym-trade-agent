"""
Targeted fix for data alignment error in calculus_strategy.py
"""

def fixed_get_latest_signal(self, symbol, prices):
    """Get calculus-based trading signal with PROPER data alignment."""
    try:
        import pandas as pd
        import numpy as np
        
        # Convert to aligned pandas Series
        if not isinstance(prices, pd.Series):
            prices = pd.Series(prices)
        
        # Remove NaN values and ensure proper alignment
        clean_prices = prices.dropna().reset_index(drop=True)
        
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
        
        # EMA smoothing with proper alignment (FIXES DATA ALIGNMENT)
        ema_prices = clean_prices.ewm(alpha=0.75).mean()
        ema_clean = ema_prices.dropna().reset_index(drop=True)
        
        # First derivative (velocity) with proper alignment
        velocity = ema_clean.diff().dropna().reset_index(drop=True)
        
        # Second derivative (acceleration) with proper alignment
        acceleration = velocity.diff().dropna().reset_index(drop=True)
        
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
        print(f"Calculus signal error for {symbol}: {e}")
        return {
            'signal': 0,
            'confidence': 0.0,
            'velocity': 0.0,
            'acceleration': 0.0,
            'snr': 0.0,
            'ema_price': None
        }

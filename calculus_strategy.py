import pandas as pd
import numpy as np
from quantitative_models import (
    smooth_price_series,
    calculate_velocity,
    calculate_acceleration,
    calculate_velocity_variance,
    taylor_expansion_forecast,
)

def generate_trading_signals(prices: pd.Series) -> pd.DataFrame:
    """
    Generates trading signals based on the calculus-based trading model.

    Args:
        prices: A pandas Series of prices.

    Returns:
        A pandas DataFrame with the trading signals.
    """
    smoothed_prices = smooth_price_series(prices)
    velocity = calculate_velocity(smoothed_prices)
    acceleration = calculate_acceleration(velocity)
    velocity_variance = calculate_velocity_variance(velocity)

    signals = pd.DataFrame(index=prices.index)
    signals['price'] = prices
    signals['smoothed_price'] = smoothed_prices
    signals['velocity'] = velocity
    signals['acceleration'] = acceleration
    signals['velocity_variance'] = velocity_variance

    # Generate trading signals
    signals['signal'] = 0
    signals.loc[(signals['velocity'] > 0) & (signals['acceleration'] > 0), 'signal'] = 1  # Buy
    signals.loc[(signals['velocity'] < 0) & (signals['acceleration'] < 0), 'signal'] = -1  # Sell
    signals.loc[(signals['velocity'].shift(1) < 0) & (signals['velocity'] > 0) & (signals['acceleration'] > 0), 'signal'] = 2 # Strong Buy (bottom forming)
    signals.loc[(signals['velocity'].shift(1) > 0) & (signals['velocity'] < 0) & (signals['acceleration'] < 0), 'signal'] = -2 # Strong Sell (top forming)

    # Generate forecasts
    signals['forecast'] = np.nan
    for i in range(1, len(signals)):
        p_hat = signals['smoothed_price'].iloc[i]
        v = signals['velocity'].iloc[i]
        a = signals['acceleration'].iloc[i]
        if not np.isnan(p_hat) and not np.isnan(v) and not np.isnan(a):
            signals['forecast'].iloc[i] = taylor_expansion_forecast(p_hat, v, a)

    return signals

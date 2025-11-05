import numpy as np
import pandas as pd

def smooth_price_series(prices: pd.Series, lambda_param: float = 0.6) -> pd.Series:
    """
    Smooths a price series using an Exponential Moving Average (EMA).

    Args:
        prices: A pandas Series of prices.
        lambda_param: The smoothing parameter for the EMA.

    Returns:
        A pandas Series of smoothed prices.
    """
    return prices.ewm(alpha=1 - lambda_param, adjust=False).mean()

def calculate_velocity(smoothed_prices: pd.Series) -> pd.Series:
    """
    Calculates the first derivative (velocity) of a smoothed price series.

    Args:
        smoothed_prices: A pandas Series of smoothed prices.

    Returns:
        A pandas Series representing the velocity of the price series.
    """
    return smoothed_prices.diff()

def calculate_acceleration(velocity: pd.Series) -> pd.Series:
    """
    Calculates the second derivative (acceleration) of a smoothed price series.

    Args:
        velocity: A pandas Series representing the velocity of the price series.

    Returns:
        A pandas Series representing the acceleration of the price series.
    """
    return velocity.diff()

def calculate_velocity_variance(velocity: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculates the variance of the velocity.

    Args:
        velocity: A pandas Series representing the velocity of the price series.
        window: The rolling window to use for the variance calculation.

    Returns:
        A pandas Series representing the variance of the velocity.
    """
    return velocity.rolling(window=window).var()

def taylor_expansion_forecast(p_hat: float, v: float, a: float, delta: int = 1) -> float:
    """
    Forecasts the next price point using a Taylor expansion.

    Args:
        p_hat: The current smoothed price.
        v: The current velocity.
        a: The current acceleration.
        delta: The time interval to forecast ahead.

    Returns:
        The forecasted price.
    """
    return p_hat + v * delta + 0.5 * a * (delta ** 2)

#!/usr/bin/env python3
"""
Sophisticated Quantitative Position Sizing and Economic Modeling System

Institutional-grade position sizing models that replace the fixed $3.00 positioning
with dynamic, risk-adjusted, and economically optimized sizing strategies.

This module implements:
1. Kelly Criterion optimal position sizing
2. Volatility-adjusted position sizing (ATR, standard deviation)
3. Correlation-based portfolio optimization
4. Account balance scaling with compound growth
5. Market psychology integration (Fear/Greed, FOMO detection)
6. Economic system dynamics (regime detection, liquidity modeling)
7. Portfolio optimization framework (risk parity, Sharpe optimization)
8. Monte Carlo simulation and backtesting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import math
from scipy import stats
from scipy.optimize import minimize
import warnings

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime classification"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    CRISIS = "crisis"

class RiskLevel(Enum):
    """Risk level classification"""
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class MarketPsychology:
    """Market psychology indicators"""
    fear_greed_index: float  # 0-100 scale
    fomo_intensity: float    # 0-100 scale
    sentiment_score: float   # -100 to +100
    social_volume: float     # Relative volume
    volatility_premium: float # Implied vs realized vol spread

@dataclass
class AssetMetrics:
    """Comprehensive asset metrics for position sizing"""
    symbol: str
    current_price: float
    volatility: float          # Annualized volatility
    atr_14d: float           # Average True Range (14 days)
    beta: float              # Market beta
    sharpe_30d: float        # 30-day Sharpe ratio
    max_dd_30d: float        # 30-day maximum drawdown
    correlation_btc: float   # Correlation with Bitcoin
    liquidity_score: float   # 0-100 liquidity metric
    funding_rate: float      # Current funding rate
    open_interest_change: float # OI change percentage

@dataclass
class PortfolioMetrics:
    """Portfolio-wide metrics"""
    total_value: float
    available_cash: float
    positions: Dict[str, float]  # symbol -> USD value
    portfolio_volatility: float
    portfolio_beta: float
    max_drawdown: float
    sharpe_ratio: float
    correlation_matrix: pd.DataFrame

class KellyCriterionSizer:
    """
    Kelly Criterion position sizing for optimal growth

    Formula: f* = (bp - q) / b
    Where:
    - f* = fraction of capital to wager
    - b = odds received on the bet (profit/loss ratio)
    - p = probability of winning
    - q = probability of losing (1 - p)

    For continuous returns: f* = μ / σ²
    Where:
    - μ = expected return
    - σ² = variance of returns
    """

    def __init__(self, kelly_fraction: float = 0.25):
        """
        Initialize Kelly Criterion sizer

        Args:
            kelly_fraction: Fraction of full Kelly to use (0.25 = Quarter Kelly)
        """
        self.kelly_fraction = kelly_fraction

    def calculate_kelly_fraction(
        self,
        expected_return: float,
        volatility: float,
        win_rate: float = None,
        profit_loss_ratio: float = None
    ) -> float:
        """
        Calculate Kelly fraction using either continuous or discrete approach

        Args:
            expected_return: Expected annual return (decimal)
            volatility: Annual volatility (decimal)
            win_rate: Win rate for discrete approach (optional)
            profit_loss_ratio: Average profit/loss ratio (optional)

        Returns:
            Kelly fraction (0-1)
        """
        if win_rate is not None and profit_loss_ratio is not None:
            # Discrete Kelly: f* = (bp - q) / b
            lose_rate = 1 - win_rate
            kelly = (win_rate * profit_loss_ratio - lose_rate) / profit_loss_ratio
        else:
            # Continuous Kelly: f* = μ / σ²
            if volatility <= 0:
                return 0.0
            kelly = expected_return / (volatility ** 2)

        # Apply fractional Kelly for safety
        safe_kelly = kelly * self.kelly_fraction

        # Boundaries: max 25% of capital, min 0
        return max(0, min(safe_kelly, 0.25))

    def calculate_position_size(
        self,
        portfolio_value: float,
        expected_return: float,
        volatility: float,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate position size using Kelly Criterion with confidence adjustment

        Args:
            portfolio_value: Total portfolio value
            expected_return: Expected return estimate
            volatility: Return volatility
            confidence_level: Confidence in expected return (0-1)

        Returns:
            Position size in USD
        """
        # Adjust expected return by confidence level
        adjusted_return = expected_return * confidence_level

        # Calculate Kelly fraction
        kelly_frac = self.calculate_kelly_fraction(adjusted_return, volatility)

        # Calculate position size
        position_size = portfolio_value * kelly_frac

        return position_size

class VolatilityAdjustedSizer:
    """
    Volatility-adjusted position sizing based on ATR and standard deviation

    Principle: Higher volatility = smaller position size to maintain consistent risk
    """

    def __init__(self, target_volatility: float = 0.15):
        """
        Initialize volatility-adjusted sizer

        Args:
            target_volatility: Target annual volatility for positions (default 15%)
        """
        self.target_volatility = target_volatility

    def calculate_atr_position_size(
        self,
        portfolio_value: float,
        asset_price: float,
        atr: float,
        risk_per_trade: float = 0.02
    ) -> float:
        """
        Calculate position size using ATR-based risk management

        Args:
            portfolio_value: Total portfolio value
            asset_price: Current asset price
            atr: Average True Range (same unit as price)
            risk_per_trade: Risk per trade as fraction of portfolio (default 2%)

        Returns:
            Position size in USD
        """
        # Calculate risk amount in USD
        risk_amount = portfolio_value * risk_per_trade

        # Calculate stop distance using 2x ATR
        stop_distance = atr * 2

        # Calculate position size based on ATR risk
        shares_by_atr = risk_amount / stop_distance
        position_size = shares_by_atr * asset_price

        return position_size

    def calculate_volatility_scaled_size(
        self,
        portfolio_value: float,
        asset_volatility: float,
        max_position_pct: float = 0.10
    ) -> float:
        """
        Scale position size based on volatility relative to target

        Args:
            portfolio_value: Total portfolio value
            asset_volatility: Asset annual volatility
            max_position_pct: Maximum position size as percentage

        Returns:
            Position size in USD
        """
        # Calculate volatility scaling factor
        vol_factor = self.target_volatility / asset_volatility

        # Bound the factor to reasonable range (0.1 to 2.0)
        vol_factor = max(0.1, min(vol_factor, 2.0))

        # Calculate base position size
        base_position = portfolio_value * max_position_pct

        # Scale by volatility factor
        scaled_position = base_position * vol_factor

        return scaled_position

class CorrelationAdjustedSizer:
    """
    Correlation-based portfolio optimization for position sizing

    Adjusts position sizes based on portfolio correlation and concentration
    """

    def __init__(self, max_portfolio_weight: float = 0.30):
        """
        Initialize correlation-adjusted sizer

        Args:
            max_portfolio_weight: Maximum weight for any single position
        """
        self.max_portfolio_weight = max_portfolio_weight

    def calculate_correlation_adjusted_size(
        self,
        base_position_size: float,
        portfolio: PortfolioMetrics,
        new_asset_correlations: Dict[str, float],
        symbol: str
    ) -> float:
        """
        Adjust position size based on portfolio correlations

        Args:
            base_position_size: Initial position size
            portfolio: Current portfolio metrics
            new_asset_correlations: Correlations with existing positions
            symbol: Symbol being added

        Returns:
            Adjusted position size in USD
        """
        if not portfolio.positions:
            return base_position_size

        # Calculate average correlation with existing positions
        correlations = list(new_asset_correlations.values())
        avg_correlation = np.mean(correlations) if correlations else 0

        # Calculate concentration penalty
        concentration_factor = 1.0

        # Penalty for high correlation (reduce position size)
        if avg_correlation > 0.7:
            concentration_factor *= 0.5
        elif avg_correlation > 0.5:
            concentration_factor *= 0.7
        elif avg_correlation > 0.3:
            concentration_factor *= 0.85

        # Penalty for portfolio concentration
        current_positions = len(portfolio.positions)
        if current_positions >= 10:
            concentration_factor *= 0.8
        elif current_positions >= 7:
            concentration_factor *= 0.9

        # Calculate maximum position based on portfolio value
        max_position = portfolio.total_value * self.max_portfolio_weight

        # Apply adjustments
        adjusted_position = base_position_size * concentration_factor
        adjusted_position = min(adjusted_position, max_position)

        return adjusted_position

class MarketPsychologySizer:
    """
    Market psychology integration for position sizing adjustments

    Adjusts positions based on Fear/Greed index, FOMO, and sentiment
    """

    def __init__(self):
        """Initialize market psychology sizer"""
        self.fear_greed_thresholds = {
            'extreme_fear': 20,
            'fear': 40,
            'neutral': 60,
            'greed': 80,
            'extreme_greed': 100
        }

    def calculate_psychology_adjustment(
        self,
        base_position_size: float,
        psychology: MarketPsychology,
        market_regime: MarketRegime
    ) -> float:
        """
        Adjust position size based on market psychology

        Args:
            base_position_size: Initial position size
            psychology: Market psychology indicators
            market_regime: Current market regime

        Returns:
            Adjusted position size in USD
        """
        adjustment_factor = 1.0

        # Fear/Greed adjustment
        fear_greed = psychology.fear_greed_index
        if fear_greed <= 20:  # Extreme fear
            adjustment_factor *= 1.3  # Increase size - contrarian opportunity
        elif fear_greed <= 40:  # Fear
            adjustment_factor *= 1.15
        elif fear_greed >= 80:  # Extreme greed
            adjustment_factor *= 0.5  # Reduce size - bubble risk
        elif fear_greed >= 60:  # Greed
            adjustment_factor *= 0.75

        # FOMO adjustment (counter-cyclical)
        fomo = psychology.fomo_intensity
        if fomo >= 80:  # Extreme FOMO
            adjustment_factor *= 0.4  # Significantly reduce
        elif fomo >= 60:  # High FOMO
            adjustment_factor *= 0.6

        # Market regime adjustment
        if market_regime == MarketRegime.CRISIS:
            adjustment_factor *= 0.2  # Very conservative in crisis
        elif market_regime == MarketRegime.BEAR:
            adjustment_factor *= 0.6  # Conservative in bear market
        elif market_regime == MarketRegime.BULL:
            adjustment_factor *= 1.1  # Slightly increase in bull market

        # Sentiment score adjustment
        sentiment = psychology.sentiment_score
        if sentiment < -50:  # Very negative sentiment
            adjustment_factor *= 1.2  # Contrarian increase
        elif sentiment > 50:  # Very positive sentiment
            adjustment_factor *= 0.8  # Reduce bullish extremes

        adjusted_position = base_position_size * adjustment_factor
        return adjusted_position

class EconomicRegimeDetector:
    """
    Economic regime detection for adaptive risk parameters

    Identifies market regimes and adjusts risk parameters accordingly
    """

    def __init__(self, lookback_days: int = 30):
        """
        Initialize regime detector

        Args:
            lookback_days: Days to look back for regime analysis
        """
        self.lookback_days = lookback_days

    def detect_market_regime(
        self,
        price_data: pd.Series,
        volume_data: pd.Series,
        volatility_data: pd.Series
    ) -> MarketRegime:
        """
        Detect current market regime

        Args:
            price_data: Historical price data
            volume_data: Historical volume data
            volatility_data: Historical volatility data

        Returns:
            Current market regime
        """
        if len(price_data) < 10:
            return MarketRegime.SIDEWAYS

        # Calculate returns
        returns = price_data.pct_change().dropna()

        # Calculate trend metrics
        recent_return = returns.tail(5).mean()
        overall_return = returns.mean()

        # Calculate volatility metrics
        recent_vol = returns.tail(5).std() * np.sqrt(252)
        avg_vol = returns.std() * np.sqrt(252)

        # Calculate volume trend
        volume_trend = volume_data.tail(5).mean() / volume_data.mean()

        # Regime classification logic
        if recent_vol > avg_vol * 2:
            return MarketRegime.VOLATILE
        elif recent_vol > avg_vol * 3:
            return MarketRegime.CRISIS
        elif recent_return > 0.02 and overall_return > 0.01:
            return MarketRegime.BULL
        elif recent_return < -0.02 and overall_return < -0.01:
            return MarketRegime.BEAR
        else:
            return MarketRegime.SIDEWAYS

    def get_regime_risk_parameters(
        self,
        regime: MarketRegime,
        base_params: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Get risk parameters adjusted for market regime

        Args:
            regime: Current market regime
            base_params: Base risk parameters

        Returns:
            Adjusted risk parameters
        """
        params = base_params.copy()

        # Adjust parameters based on regime
        if regime == MarketRegime.CRISIS:
            params['max_leverage'] *= 0.2
            params['max_position_size'] *= 0.3
            params['stop_loss_pct'] *= 0.5
        elif regime == MarketRegime.BEAR:
            params['max_leverage'] *= 0.5
            params['max_position_size'] *= 0.6
            params['stop_loss_pct'] *= 0.7
        elif regime == MarketRegime.VOLATILE:
            params['max_leverage'] *= 0.4
            params['max_position_size'] *= 0.7
            params['stop_loss_pct'] *= 0.8
        elif regime == MarketRegime.BULL:
            params['max_leverage'] *= 1.1
            params['max_position_size'] *= 1.1
            params['stop_loss_pct'] *= 1.1

        return params

class CompoundGrowthOptimizer:
    """
    Compound growth optimization for position scaling

    Optimizes position sizing for maximum compound growth over time
    """

    def __init__(self, target_annual_return: float = 0.50):
        """
        Initialize compound growth optimizer

        Args:
            target_annual_return: Target annual return (default 50%)
        """
        self.target_annual_return = target_annual_return

    def calculate_optimal_reinvestment_rate(
        self,
        current_portfolio_value: float,
        historical_returns: List[float],
        volatility: float
    ) -> float:
        """
        Calculate optimal reinvestment rate using growth optimization

        Args:
            current_portfolio_value: Current portfolio value
            historical_returns: Historical return series
            volatility: Current volatility estimate

        Returns:
            Optimal reinvestment rate (0-1)
        """
        if len(historical_returns) < 10:
            return 0.5  # Conservative default

        # Calculate historical metrics
        returns = np.array(historical_returns)
        mean_return = np.mean(returns)
        return_std = np.std(returns)

        # Calculate Sharpe ratio
        sharpe = mean_return / return_std if return_std > 0 else 0

        # Optimize reinvestment rate using Kelly
        if return_std > 0:
            kelly_rate = mean_return / (return_std ** 2)
            # Apply fractional Kelly for safety
            optimal_rate = kelly_rate * 0.25
        else:
            optimal_rate = 0.5

        # Boundaries
        optimal_rate = max(0.1, min(optimal_rate, 0.8))

        return optimal_rate

    def calculate_compound_position_size(
        self,
        base_position_size: float,
        portfolio_value: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate position size optimized for compound growth

        Args:
            base_position_size: Base position size
            portfolio_value: Total portfolio value
            win_rate: Historical win rate
            avg_win: Average win amount
            avg_loss: Average loss amount

        Returns:
            Compound-optimized position size
        """
        # Calculate expected value
        expected_value = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        # If positive EV, can be more aggressive
        if expected_value > 0:
            # Calculate optimal fraction using Kelly
            win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1
            kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio

            # Apply fractional Kelly
            safe_fraction = kelly_fraction * 0.25

            # Calculate compound position
            compound_position = portfolio_value * safe_fraction

            # Don't exceed reasonable multiple of base position
            max_position = base_position_size * 2
            compound_position = min(compound_position, max_position)
        else:
            # Negative or zero EV - use conservative sizing
            compound_position = base_position_size * 0.5

        return compound_position

class QuantitativePositionSizer:
    """
    Main quantitative position sizing system that integrates all models

    This is the primary interface that combines all sophisticated sizing models
    into institutional-grade position sizing recommendations.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the quantitative position sizer

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Initialize component models
        self.kelly_sizer = KellyCriterionSizer(
            kelly_fraction=self.config.get('kelly_fraction', 0.25)
        )
        self.volatility_sizer = VolatilityAdjustedSizer(
            target_volatility=self.config.get('target_volatility', 0.15)
        )
        self.correlation_sizer = CorrelationAdjustedSizer(
            max_portfolio_weight=self.config.get('max_portfolio_weight', 0.30)
        )
        self.psychology_sizer = MarketPsychologySizer()
        self.regime_detector = EconomicRegimeDetector(
            lookback_days=self.config.get('lookback_days', 30)
        )
        self.compound_optimizer = CompoundGrowthOptimizer(
            target_annual_return=self.config.get('target_annual_return', 0.50)
        )

        # Risk parameters by regime
        self.base_risk_params = {
            'max_leverage': 10.0,
            'max_position_size': 0.10,  # 10% of portfolio
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.15
        }

        logger.info("Quantitative Position Sizer initialized with institutional-grade models")

    def calculate_optimal_position_size(
        self,
        symbol: str,
        asset_metrics: AssetMetrics,
        portfolio: PortfolioMetrics,
        market_psychology: MarketPsychology,
        historical_data: Dict[str, pd.Series],
        confidence_level: float = 0.8
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate optimal position size using all quantitative models

        Args:
            symbol: Asset symbol
            asset_metrics: Comprehensive asset metrics
            portfolio: Current portfolio metrics
            market_psychology: Market psychology indicators
            historical_data: Historical price/volume/volatility data
            confidence_level: Confidence in return estimates (0-1)

        Returns:
            Tuple of (optimal_position_size_usd, sizing_details)
        """
        try:
            # Detect market regime
            market_regime = self.regime_detector.detect_market_regime(
                historical_data['price'],
                historical_data['volume'],
                historical_data['volatility']
            )

            # Get regime-adjusted risk parameters
            risk_params = self.regime_detector.get_regime_risk_parameters(
                market_regime, self.base_risk_params
            )

            # Calculate base position sizes using different models
            sizing_results = {}

            # 1. Kelly Criterion sizing
            kelly_size = self.kelly_sizer.calculate_position_size(
                portfolio.total_value,
                asset_metrics.sharpe_30d * asset_metrics.volatility,  # Expected return proxy
                asset_metrics.volatility,
                confidence_level
            )
            sizing_results['kelly_size'] = kelly_size

            # 2. Volatility-adjusted sizing
            atr_size = self.volatility_sizer.calculate_atr_position_size(
                portfolio.total_value,
                asset_metrics.current_price,
                asset_metrics.atr_14d
            )
            sizing_results['atr_size'] = atr_size

            vol_scaled_size = self.volatility_sizer.calculate_volatility_scaled_size(
                portfolio.total_value,
                asset_metrics.volatility,
                risk_params['max_position_size']
            )
            sizing_results['volatility_scaled'] = vol_scaled_size

            # 3. Calculate correlations with existing positions
            correlations = {}
            for existing_symbol in portfolio.positions.keys():
                if existing_symbol != symbol:
                    # Use asset correlation with BTC as proxy for crypto correlations
                    correlations[existing_symbol] = asset_metrics.correlation_btc

            # 4. Base position selection (use minimum of conservative estimates)
            base_sizes = [kelly_size, atr_size, vol_scaled_size]
            base_position = min(base_sizes)  # Conservative approach
            sizing_results['base_position'] = base_position

            # 5. Apply correlation adjustment
            correlation_adjusted = self.correlation_sizer.calculate_correlation_adjusted_size(
                base_position,
                portfolio,
                correlations,
                symbol
            )
            sizing_results['correlation_adjusted'] = correlation_adjusted

            # 6. Apply psychology adjustment
            psychology_adjusted = self.psychology_sizer.calculate_psychology_adjustment(
                correlation_adjusted,
                market_psychology,
                market_regime
            )
            sizing_results['psychology_adjusted'] = psychology_adjusted

            # 7. Apply compound growth optimization
            # Estimate historical performance metrics
            win_rate = max(0.4, min(0.7, 0.5 + asset_metrics.sharpe_30d * 0.1))
            avg_win = asset_metrics.current_price * asset_metrics.volatility * 2
            avg_loss = asset_metrics.current_price * asset_metrics.volatility

            compound_adjusted = self.compound_optimizer.calculate_compound_position_size(
                psychology_adjusted,
                portfolio.total_value,
                win_rate,
                avg_win,
                avg_loss
            )
            sizing_results['compound_adjusted'] = compound_adjusted

            # 8. Apply risk limits
            max_position = portfolio.total_value * risk_params['max_position_size']
            min_position = portfolio.total_value * 0.01  # Minimum 1% position

            final_position = max(min_position, min(compound_adjusted, max_position))
            sizing_results['final_position'] = final_position

            # Calculate recommended leverage
            risk_per_dollar = asset_metrics.volatility / np.sqrt(252)  # Daily volatility
            max_acceptable_risk = 0.02  # 2% max daily risk
            recommended_leverage = max_acceptable_risk / risk_per_dollar
            recommended_leverage = min(recommended_leverage, risk_params['max_leverage'])
            recommended_leverage = max(1.0, recommended_leverage)  # Minimum 1x

            sizing_results.update({
                'recommended_leverage': recommended_leverage,
                'market_regime': market_regime.value,
                'risk_params': risk_params,
                'confidence_level': confidence_level,
                'position_risk_pct': final_position / portfolio.total_value
            })

            logger.info(f"Optimal position calculated for {symbol}: ${final_position:.2f}")
            logger.info(f"   Market regime: {market_regime.value}")
            logger.info(f"   Recommended leverage: {recommended_leverage:.1f}x")
            logger.info(f"   Position risk: {sizing_results['position_risk_pct']*100:.1f}%")

            return final_position, sizing_results

        except Exception as e:
            logger.error(f"Error calculating optimal position size for {symbol}: {str(e)}")
            # Return conservative fallback
            fallback_size = portfolio.total_value * 0.02  # 2% conservative position
            return fallback_size, {'error': str(e), 'fallback': True}

# Monte Carlo simulation for position sizing validation
class PositionSizeValidator:
    """
    Monte Carlo simulation for validating position sizing strategies

    Tests position sizing models under various market conditions
    """

    def __init__(self, num_simulations: int = 10000):
        """
        Initialize position size validator

        Args:
            num_simulations: Number of Monte Carlo simulations
        """
        self.num_simulations = num_simulations

    def validate_position_sizing(
        self,
        initial_capital: float,
        position_size_frac: float,
        expected_return: float,
        volatility: float,
        num_trades: int = 252
    ) -> Dict[str, float]:
        """
        Validate position sizing using Monte Carlo simulation

        Args:
            initial_capital: Starting capital
            position_size_frac: Position size as fraction of capital
            expected_return: Expected return per trade
            volatility: Return volatility per trade
            num_trades: Number of trades to simulate

        Returns:
            Dictionary with simulation results
        """
        np.random.seed(42)  # For reproducible results

        # Generate random returns
        returns = np.random.normal(expected_return, volatility, (self.num_simulations, num_trades))

        # Simulate portfolio evolution
        final_values = []
        max_drawdowns = []

        for sim_returns in returns:
            portfolio_value = initial_capital
            peak_value = initial_capital
            max_drawdown = 0

            for trade_return in sim_returns:
                # Calculate position size
                position_value = portfolio_value * position_size_frac

                # Calculate P&L for this trade
                pnl = position_value * trade_return

                # Update portfolio
                portfolio_value += pnl

                # Track drawdown
                if portfolio_value > peak_value:
                    peak_value = portfolio_value
                drawdown = (peak_value - portfolio_value) / peak_value
                max_drawdown = max(max_drawdown, drawdown)

            final_values.append(portfolio_value)
            max_drawdowns.append(max_drawdown)

        # Calculate statistics
        final_values = np.array(final_values)
        max_drawdowns = np.array(max_drawdowns)

        results = {
            'mean_final_value': np.mean(final_values),
            'median_final_value': np.median(final_values),
            'percentile_5': np.percentile(final_values, 5),
            'percentile_95': np.percentile(final_values, 95),
            'probability_of_loss': np.mean(final_values < initial_capital),
            'mean_max_drawdown': np.mean(max_drawdowns),
            'median_max_drawdown': np.median(max_drawdowns),
            'probability_of_20pct_drawdown': np.mean(max_drawdowns > 0.20),
            'probability_of_50pct_drawdown': np.mean(max_drawdowns > 0.50)
        }

        return results

if __name__ == "__main__":
    # Example usage
    sizer = QuantitativePositionSizer()

    # Create example asset metrics
    asset_metrics = AssetMetrics(
        symbol="BTCUSDT",
        current_price=50000.0,
        volatility=0.8,  # 80% annual volatility
        atr_14d=2000.0,
        beta=1.0,
        sharpe_30d=1.5,
        max_dd_30d=-0.15,
        correlation_btc=1.0,
        liquidity_score=95.0,
        funding_rate=0.0001,
        open_interest_change=0.05
    )

    # Create example portfolio
    portfolio = PortfolioMetrics(
        total_value=10000.0,
        available_cash=8000.0,
        positions={"ETHUSDT": 2000.0},
        portfolio_volatility=0.6,
        portfolio_beta=0.9,
        max_drawdown=-0.10,
        sharpe_ratio=1.2,
        correlation_matrix=pd.DataFrame()
    )

    # Create market psychology
    psychology = MarketPsychology(
        fear_greed_index=45.0,
        fomo_intensity=60.0,
        sentiment_score=20.0,
        social_volume=1.2,
        volatility_premium=0.1
    )

    # Create historical data
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    price_data = pd.Series(50000 + np.random.randn(100).cumsum() * 500, index=dates)
    volume_data = pd.Series(1000 + np.random.rand(100) * 500, index=dates)
    volatility_data = pd.Series(0.02 + np.random.rand(100) * 0.03, index=dates)

    historical_data = {
        'price': price_data,
        'volume': volume_data,
        'volatility': volatility_data
    }

    # Calculate optimal position
    optimal_size, details = sizer.calculate_optimal_position_size(
        "BTCUSDT",
        asset_metrics,
        portfolio,
        psychology,
        historical_data
    )

    print(f"Optimal position size: ${optimal_size:.2f}")
    print(f"Recommended leverage: {details['recommended_leverage']:.1f}x")
    print(f"Market regime: {details['market_regime']}")
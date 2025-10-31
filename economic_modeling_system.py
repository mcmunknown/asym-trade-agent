#!/usr/bin/env python3
"""
Advanced Economic Modeling System for Cryptocurrency Markets

Institutional-grade economic modeling that provides sophisticated analysis of:
1. Market microstructure and liquidity dynamics
2. Macroeconomic factor integration
3. Behavioral finance models for crypto markets
4. Risk-adjusted performance optimization
5. Portfolio construction and optimization algorithms
6. Market regime detection and adaptation
7. Economic scenario analysis and stress testing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
from scipy import stats, optimize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings

logger = logging.getLogger(__name__)

class LiquidityRegime(Enum):
    """Liquidity regime classification"""
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    CRITICAL = "critical"

class MarketMicrostructure:
    """
    Advanced market microstructure analysis for cryptocurrency markets

    Analyzes order flow, liquidity dynamics, and price impact
    """

    def __init__(self):
        """Initialize market microstructure analyzer"""
        self.liquidity_history = []
        self.spread_history = []
        self.depth_history = []

    def calculate_liquidity_metrics(
        self,
        order_book: Dict[str, List[Tuple[float, float]]],
        recent_trades: List[Dict],
        symbol: str
    ) -> Dict[str, float]:
        """
        Calculate comprehensive liquidity metrics

        Args:
            order_book: Current order book (bids/asks)
            recent_trades: Recent trade history
            symbol: Trading symbol

        Returns:
            Dictionary of liquidity metrics
        """
        try:
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])

            if not bids or not asks:
                return {'liquidity_score': 0.0}

            # Calculate bid-ask spread
            best_bid = bids[0][0] if bids else 0
            best_ask = asks[0][0] if asks else 0
            mid_price = (best_bid + best_ask) / 2
            spread_pct = (best_ask - best_bid) / mid_price if mid_price > 0 else 1.0

            # Calculate market depth (liquidity at different levels)
            depth_levels = [0.1, 0.5, 1.0, 2.0]  # Percentage from mid price
            depth_metrics = {}

            for level in depth_levels:
                bid_depth = sum(
                    qty for price, qty in bids
                    if abs(price - mid_price) / mid_price <= level
                )
                ask_depth = sum(
                    qty for price, qty in asks
                    if abs(price - mid_price) / mid_price <= level
                )
                depth_metrics[f'depth_{level}pct'] = (bid_depth + ask_depth) / 2

            # Calculate order book imbalance
            total_bid_qty = sum(qty for _, qty in bids[:10])  # Top 10 levels
            total_ask_qty = sum(qty for _, qty in asks[:10])
            imbalance = (total_bid_qty - total_ask_qty) / (total_bid_qty + total_ask_qty)

            # Calculate trade flow metrics
            if recent_trades:
                trade_volumes = [trade.get('qty', 0) * trade.get('price', 0) for trade in recent_trades]
                avg_trade_size = np.mean(trade_volumes)
                trade_frequency = len(recent_trades) / 60  # Trades per minute
                volume_imbalance = self._calculate_volume_imbalance(recent_trades)
            else:
                avg_trade_size = 0
                trade_frequency = 0
                volume_imbalance = 0

            # Calculate liquidity score (0-100)
            liquidity_score = self._calculate_liquidity_score(
                spread_pct, depth_metrics, trade_frequency, avg_trade_size
            )

            metrics = {
                'liquidity_score': liquidity_score,
                'spread_pct': spread_pct,
                'order_imbalance': imbalance,
                'avg_trade_size': avg_trade_size,
                'trade_frequency': trade_frequency,
                'volume_imbalance': volume_imbalance,
                **depth_metrics
            }

            # Store history for trend analysis
            self.liquidity_history.append(metrics)
            if len(self.liquidity_history) > 1000:
                self.liquidity_history.pop(0)

            return metrics

        except Exception as e:
            logger.error(f"Error calculating liquidity metrics for {symbol}: {str(e)}")
            return {'liquidity_score': 0.0}

    def _calculate_liquidity_score(
        self,
        spread_pct: float,
        depth_metrics: Dict[str, float],
        trade_frequency: float,
        avg_trade_size: float
    ) -> float:
        """Calculate overall liquidity score (0-100)"""
        # Normalize components
        spread_score = max(0, 100 - spread_pct * 10000)  # Lower spread = higher score
        depth_score = min(100, depth_metrics.get('depth_0.5pct', 0) / 1000 * 100)
        frequency_score = min(100, trade_frequency * 10)
        size_score = min(100, avg_trade_size / 10000 * 100)

        # Weighted average
        liquidity_score = (
            spread_score * 0.3 +
            depth_score * 0.3 +
            frequency_score * 0.2 +
            size_score * 0.2
        )

        return liquidity_score

    def _calculate_volume_imbalance(self, trades: List[Dict]) -> float:
        """Calculate buy/sell volume imbalance"""
        buy_volume = sum(
            trade.get('qty', 0) * trade.get('price', 0)
            for trade in trades if trade.get('side') == 'Buy'
        )
        sell_volume = sum(
            trade.get('qty', 0) * trade.get('price', 0)
            for trade in trades if trade.get('side') == 'Sell'
        )

        total_volume = buy_volume + sell_volume
        if total_volume == 0:
            return 0

        return (buy_volume - sell_volume) / total_volume

    def estimate_price_impact(
        self,
        order_size: float,
        current_price: float,
        liquidity_metrics: Dict[str, float]
    ) -> float:
        """
        Estimate price impact for a given order size

        Args:
            order_size: Order size in USD
            current_price: Current asset price
            liquidity_metrics: Liquidity metrics from calculate_liquidity_metrics

        Returns:
            Estimated price impact as percentage
        """
        liquidity_score = liquidity_metrics.get('liquidity_score', 50)
        spread_pct = liquidity_metrics.get('spread_pct', 0.001)
        depth_1pct = liquidity_metrics.get('depth_1pct', 10000)

        # Kyle's lambda: Price impact per unit volume
        # Higher liquidity = lower impact
        if depth_1pct > 0:
            kyle_lambda = spread_pct / (2 * depth_1pct)
        else:
            kyle_lambda = spread_pct / 1000

        # Calculate price impact
        price_impact = kyle_lambda * order_size

        # Adjust for liquidity
        liquidity_multiplier = 100 / liquidity_score  # Lower liquidity = higher impact
        price_impact *= liquidity_multiplier

        # Cap at reasonable levels
        price_impact = min(price_impact, 0.05)  # Max 5% impact

        return price_impact

class MacroeconomicAnalyzer:
    """
    Macroeconomic factor integration for cryptocurrency markets

    Analyzes how traditional macro factors affect crypto markets
    """

    def __init__(self):
        """Initialize macroeconomic analyzer"""
        self.factor_weights = {
            'interest_rates': 0.25,
            'inflation': 0.20,
            'risk_sentiment': 0.25,
            'dollar_strength': 0.15,
            'market_volatility': 0.15
        }

    def calculate_macro_score(
        self,
        interest_rate_change: float,
        inflation_rate: float,
        risk_sentiment: float,  # VIX-like index
        dollar_index: float,
        market_volatility: float
    ) -> Dict[str, float]:
        """
        Calculate macroeconomic impact score

        Args:
            interest_rate_change: Change in interest rates (bps)
            inflation_rate: Current inflation rate
            risk_sentiment: Risk sentiment index (0-100, lower = more fear)
            dollar_index: US Dollar index value
            market_volatility: Market volatility index

        Returns:
            Dictionary with macro scores and overall assessment
        """
        # Normalize individual factors
        interest_score = max(-100, min(100, -interest_rate_change * 10))  # Higher rates = negative for crypto
        inflation_score = max(-100, min(100, (inflation_rate - 0.02) * 1000))  # Higher inflation = positive for crypto
        sentiment_score = 100 - risk_sentiment  # Lower fear = higher score
        dollar_score = 100 - (dollar_index - 100)  # Weaker dollar = positive for crypto
        volatility_score = 100 - market_volatility  # Lower volatility = positive

        # Calculate weighted overall score
        overall_score = (
            interest_score * self.factor_weights['interest_rates'] +
            inflation_score * self.factor_weights['inflation'] +
            sentiment_score * self.factor_weights['risk_sentiment'] +
            dollar_score * self.factor_weights['dollar_strength'] +
            volatility_score * self.factor_weights['market_volatility']
        )

        # Classify regime
        if overall_score > 50:
            macro_regime = "favorable"
        elif overall_score > 20:
            macro_regime = "neutral"
        else:
            macro_regime = "unfavorable"

        return {
            'overall_score': overall_score,
            'macro_regime': macro_regime,
            'interest_rate_impact': interest_score,
            'inflation_impact': inflation_score,
            'sentiment_impact': sentiment_score,
            'dollar_impact': dollar_score,
            'volatility_impact': volatility_score
        }

class BehavioralFinanceModel:
    """
    Behavioral finance models specific to cryptocurrency markets

    Models investor psychology, herding behavior, and market sentiment
    """

    def __init__(self):
        """Initialize behavioral finance model"""
        self.sentiment_history = []
        self.fomo_indicators = []

    def calculate_fear_greed_index(
        self,
        price_momentum: float,
        volume_momentum: float,
        social_sentiment: float,
        volatility: float,
        market_dominance: Dict[str, float]
    ) -> float:
        """
        Calculate Fear & Greed Index (0-100 scale)

        Args:
            price_momentum: Recent price change percentage
            volume_momentum: Recent volume change percentage
            social_sentiment: Social media sentiment (-100 to +100)
            volatility: Current volatility (annualized)
            market_dominance: BTC dominance and other metrics

        Returns:
            Fear & Greed Index (0 = Extreme Fear, 100 = Extreme Greed)
        """
        # Momentum component (40% weight)
        momentum_score = np.clip(price_momentum * 500 + 50, 0, 100)

        # Volume component (20% weight)
        volume_score = np.clip(volume_momentum * 200 + 50, 0, 100)

        # Social sentiment component (20% weight)
        sentiment_score = np.clip(social_sentiment + 50, 0, 100)

        # Volatility component (10% weight) - inverse relationship
        volatility_score = np.clip(100 - volatility * 100, 0, 100)

        # Market dominance component (10% weight)
        btc_dominance = market_dominance.get('btc', 50)
        dominance_score = np.clip(100 - abs(btc_dominance - 60) * 2, 0, 100)

        # Calculate weighted average
        fear_greed_index = (
            momentum_score * 0.40 +
            volume_score * 0.20 +
            sentiment_score * 0.20 +
            volatility_score * 0.10 +
            dominance_score * 0.10
        )

        return fear_greed_index

    def detect_fomo_signals(
        self,
        price_data: pd.Series,
        volume_data: pd.Series,
        social_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Detect FOMO (Fear Of Missing Out) signals

        Args:
            price_data: Historical price data
            volume_data: Historical volume data
            social_metrics: Social media metrics

        Returns:
            Dictionary with FOMO indicators
        """
        if len(price_data) < 20:
            return {'fomo_intensity': 0.0}

        # Price acceleration
        returns = price_data.pct_change().dropna()
        recent_returns = returns.tail(5)
        price_acceleration = recent_returns.mean() / returns.std() if returns.std() > 0 else 0

        # Volume surge
        recent_volume = volume_data.tail(5)
        avg_volume = volume_data.tail(20).mean()
        volume_surge = recent_volume.mean() / avg_volume if avg_volume > 0 else 1

        # Social media explosion
        social_volume = social_metrics.get('volume_change', 0)
        social_sentiment = social_metrics.get('sentiment_change', 0)

        # New investor influx (proxy via search interest, etc.)
        new_interest = social_metrics.get('search_interest', 0)

        # Calculate FOMO intensity (0-100)
        fomo_components = [
            np.clip(price_acceleration * 20 + 50, 0, 100),
            np.clip(volume_surge * 25, 0, 100),
            np.clip(social_volume + 50, 0, 100),
            np.clip(social_sentiment + 50, 0, 100),
            np.clip(new_interest * 50, 0, 100)
        ]

        fomo_intensity = np.mean(fomo_components)

        # Detect herding behavior
        herding_indicator = self._detect_herding_behavior(price_data, volume_data)

        return {
            'fomo_intensity': fomo_intensity,
            'price_acceleration': price_acceleration,
            'volume_surge': volume_surge,
            'social_volume': social_volume,
            'herding_indicator': herding_indicator
        }

    def _detect_herding_behavior(
        self,
        price_data: pd.Series,
        volume_data: pd.Series
    ) -> float:
        """Detect herding behavior in market data"""
        if len(price_data) < 30:
            return 0.0

        # Calculate cross-sectional correlation (simplified for single asset)
        returns = price_data.pct_change().dropna()
        volume_returns = volume_data.pct_change().dropna()

        # Align and calculate correlation
        aligned_data = pd.concat([returns, volume_returns], axis=1).dropna()
        if len(aligned_data) < 10:
            return 0.0

        correlation = aligned_data.iloc[:, 0].corr(aligned_data.iloc[:, 1])

        # High correlation between price and volume suggests herding
        herding_strength = abs(correlation) * 100

        return herding_strength

class PortfolioOptimizer:
    """
    Advanced portfolio optimization using modern portfolio theory

    Implements mean-variance optimization, risk parity, and other strategies
    """

    def __init__(self):
        """Initialize portfolio optimizer"""
        self.optimization_methods = [
            'mean_variance',
            'risk_parity',
            'equal_weight',
            'max_sharpe',
            'min_variance'
        ]

    def optimize_portfolio(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        risk_free_rate: float = 0.02,
        method: str = 'max_sharpe',
        constraints: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Optimize portfolio weights using specified method

        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of returns
            risk_free_rate: Risk-free rate
            method: Optimization method
            constraints: Portfolio constraints

        Returns:
            Dictionary with optimization results
        """
        n_assets = len(expected_returns)

        # Default constraints
        if constraints is None:
            constraints = {
                'weight_bounds': (0.0, 1.0),  # No short selling
                'max_weight': 0.4,  # Max 40% in any asset
                'min_weight': 0.05  # Min 5% in any asset
            }

        # Initial guess (equal weights)
        initial_weights = np.array([1.0 / n_assets] * n_assets)

        if method == 'equal_weight':
            weights = initial_weights
        elif method == 'risk_parity':
            weights = self._risk_parity_optimization(covariance_matrix)
        else:
            # Use scipy optimization for mean-variance methods
            weights = self._mean_variance_optimization(
                expected_returns,
                covariance_matrix,
                risk_free_rate,
                method,
                constraints,
                initial_weights
            )

        # Calculate portfolio metrics
        portfolio_return = np.sum(weights * expected_returns)
        portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

        results = {
            'weights': pd.Series(weights, index=expected_returns.index),
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'method': method
        }

        return results

    def _risk_parity_optimization(self, covariance_matrix: pd.DataFrame) -> np.ndarray:
        """Implement risk parity optimization"""
        n_assets = len(covariance_matrix)

        def risk_parity_objective(weights):
            portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
            marginal_contrib = np.dot(covariance_matrix, weights)
            contrib = weights * marginal_contrib / portfolio_variance
            return np.sum((contrib - 1/n_assets) ** 2)

        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        )
        bounds = tuple((0.0, 1.0) for _ in range(n_assets))

        result = optimize.minimize(
            risk_parity_objective,
            x0=np.array([1.0/n_assets] * n_assets),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        return result.x if result.success else np.array([1.0/n_assets] * n_assets)

    def _mean_variance_optimization(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        risk_free_rate: float,
        method: str,
        constraints: Dict,
        initial_weights: np.ndarray
    ) -> np.ndarray:
        """Implement mean-variance optimization"""
        n_assets = len(expected_returns)

        def negative_sharpe_ratio(weights):
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            return -(portfolio_return - risk_free_rate) / portfolio_volatility

        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(covariance_matrix, weights))

        def negative_return(weights):
            return -np.sum(weights * expected_returns)

        # Select objective function
        if method == 'max_sharpe':
            objective = negative_sharpe_ratio
        elif method == 'min_variance':
            objective = portfolio_variance
        elif method == 'max_return':
            objective = negative_return
        else:
            objective = negative_sharpe_ratio

        # Set up constraints
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]

        # Add max weight constraint
        max_weight = constraints.get('max_weight', 0.4)
        min_weight = constraints.get('min_weight', 0.0)
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))

        result = optimize.minimize(
            objective,
            x0=initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )

        return result.x if result.success else initial_weights

class EconomicModelingSystem:
    """
    Main economic modeling system that integrates all components

    Provides comprehensive economic analysis for cryptocurrency trading
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize economic modeling system

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Initialize components
        self.microstructure = MarketMicrostructure()
        self.macro_analyzer = MacroeconomicAnalyzer()
        self.behavioral_model = BehavioralFinanceModel()
        self.portfolio_optimizer = PortfolioOptimizer()

        # Data storage
        self.market_data_history = {}
        self.analysis_history = []

        logger.info("Economic Modeling System initialized with institutional-grade components")

    def analyze_market_conditions(
        self,
        symbol: str,
        order_book: Dict[str, List[Tuple[float, float]]],
        recent_trades: List[Dict],
        price_history: pd.Series,
        volume_history: pd.Series,
        macro_data: Dict[str, float],
        social_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Comprehensive market condition analysis

        Args:
            symbol: Trading symbol
            order_book: Current order book
            recent_trades: Recent trade history
            price_history: Historical price data
            volume_history: Historical volume data
            macro_data: Macroeconomic data
            social_metrics: Social media metrics

        Returns:
            Comprehensive market analysis
        """
        try:
            analysis = {}

            # 1. Market microstructure analysis
            liquidity_metrics = self.microstructure.calculate_liquidity_metrics(
                order_book, recent_trades, symbol
            )
            analysis['liquidity'] = liquidity_metrics

            # 2. Macroeconomic analysis
            macro_score = self.macro_analyzer.calculate_macro_score(**macro_data)
            analysis['macro'] = macro_score

            # 3. Behavioral finance analysis
            fear_greed = self.behavioral_model.calculate_fear_greed_index(
                price_momentum=price_history.pct_change().tail(7).sum(),
                volume_momentum=volume_history.pct_change().tail(7).sum(),
                social_sentiment=social_metrics.get('sentiment', 0),
                volatility=price_history.pct_change().std() * np.sqrt(252),
                market_dominance=social_metrics.get('dominance', {'btc': 50})
            )
            analysis['fear_greed'] = fear_greed

            fomo_signals = self.behavioral_model.detect_fomo_signals(
                price_history, volume_history, social_metrics
            )
            analysis['fomo'] = fomo_signals

            # 4. Market regime classification
            regime = self._classify_market_regime(analysis)
            analysis['regime'] = regime

            # 5. Trading recommendations
            recommendations = self._generate_trading_recommendations(analysis)
            analysis['recommendations'] = recommendations

            # 6. Risk assessment
            risk_assessment = self._assess_market_risk(analysis)
            analysis['risk'] = risk_assessment

            # Store analysis
            self.analysis_history.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'analysis': analysis
            })

            if len(self.analysis_history) > 1000:
                self.analysis_history.pop(0)

            logger.info(f"Market analysis completed for {symbol}")
            logger.info(f"   Regime: {regime}")
            logger.info(f"   Fear & Greed: {fear_greed:.1f}")
            logger.info(f"   Liquidity Score: {liquidity_metrics.get('liquidity_score', 0):.1f}")

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing market conditions for {symbol}: {str(e)}")
            return {'error': str(e)}

    def _classify_market_regime(self, analysis: Dict) -> str:
        """Classify current market regime based on all indicators"""
        fear_greed = analysis.get('fear_greed', 50)
        liquidity = analysis.get('liquidity', {}).get('liquidity_score', 50)
        macro_score = analysis.get('macro', {}).get('overall_score', 50)
        fomo_intensity = analysis.get('fomo', {}).get('fomo_intensity', 0)

        # Regime classification logic
        if fear_greed < 25 and liquidity < 30:
            return "CRISIS"
        elif fear_greed < 35 and macro_score < 30:
            return "BEAR"
        elif fomo_intensity > 80 and fear_greed > 75:
            return "EUPHORIA"
        elif fear_greed > 65 and liquidity > 70:
            return "BULL"
        elif 35 <= fear_greed <= 65 and 40 <= liquidity <= 70:
            return "NORMAL"
        else:
            return "TRANSITION"

    def _generate_trading_recommendations(self, analysis: Dict) -> Dict[str, Any]:
        """Generate trading recommendations based on analysis"""
        regime = analysis.get('regime', 'NORMAL')
        fear_greed = analysis.get('fear_greed', 50)
        liquidity = analysis.get('liquidity', {}).get('liquidity_score', 50)

        recommendations = {
            'position_sizing_multiplier': 1.0,
            'leverage_recommendation': 1.0,
            'risk_adjustment': 'normal',
            'trading_bias': 'neutral',
            'liquidity_adjustment': 1.0
        }

        # Adjust based on regime
        if regime == "CRISIS":
            recommendations.update({
                'position_sizing_multiplier': 0.2,
                'leverage_recommendation': 1.0,
                'risk_adjustment': 'conservative',
                'trading_bias': 'defensive'
            })
        elif regime == "BEAR":
            recommendations.update({
                'position_sizing_multiplier': 0.5,
                'leverage_recommendation': 2.0,
                'risk_adjustment': 'conservative',
                'trading_bias': 'cautious'
            })
        elif regime == "BULL":
            recommendations.update({
                'position_sizing_multiplier': 1.2,
                'leverage_recommendation': 5.0,
                'risk_adjustment': 'moderate',
                'trading_bias': 'aggressive'
            })
        elif regime == "EUPHORIA":
            recommendations.update({
                'position_sizing_multiplier': 0.6,  # Reduce in euphoria (bubble risk)
                'leverage_recommendation': 3.0,
                'risk_adjustment': 'conservative',
                'trading_bias': 'contrarian'
            })

        # Adjust for liquidity
        if liquidity < 30:
            recommendations['liquidity_adjustment'] = 0.5
            recommendations['position_sizing_multiplier'] *= 0.5
        elif liquidity < 50:
            recommendations['liquidity_adjustment'] = 0.8
            recommendations['position_sizing_multiplier'] *= 0.8

        return recommendations

    def _assess_market_risk(self, analysis: Dict) -> Dict[str, float]:
        """Assess overall market risk"""
        fear_greed = analysis.get('fear_greed', 50)
        liquidity = analysis.get('liquidity', {}).get('liquidity_score', 50)
        fomo_intensity = analysis.get('fomo', {}).get('fomo_intensity', 0)
        macro_score = analysis.get('macro', {}).get('overall_score', 50)

        # Calculate risk components
        volatility_risk = max(0, (100 - liquidity) / 100)
        psychology_risk = max(0, fomo_intensity / 100)
        macro_risk = max(0, (100 - macro_score) / 100)
        sentiment_risk = max(0, (100 - fear_greed) / 100) if fear_greed < 50 else 0

        # Overall risk score
        overall_risk = (
            volatility_risk * 0.3 +
            psychology_risk * 0.25 +
            macro_risk * 0.25 +
            sentiment_risk * 0.2
        )

        return {
            'overall_risk_score': overall_risk,
            'volatility_risk': volatility_risk,
            'psychology_risk': psychology_risk,
            'macro_risk': macro_risk,
            'sentiment_risk': sentiment_risk
        }

    def optimize_crypto_portfolio(
        self,
        assets: List[str],
        expected_returns: Dict[str, float],
        return_data: pd.DataFrame,
        optimization_method: str = 'risk_parity'
    ) -> Dict[str, Any]:
        """
        Optimize cryptocurrency portfolio allocation

        Args:
            assets: List of cryptocurrency assets
            expected_returns: Expected returns for each asset
            return_data: Historical return data for assets
            optimization_method: Portfolio optimization method

        Returns:
            Portfolio optimization results
        """
        try:
            # Prepare data
            returns_df = return_data[assets].dropna()
            expected_returns_series = pd.Series([expected_returns.get(asset, 0) for asset in assets], index=assets)
            covariance_matrix = returns_df.cov() * 252  # Annualized covariance

            # Optimize portfolio
            optimization_results = self.portfolio_optimizer.optimize_portfolio(
                expected_returns_series,
                covariance_matrix,
                method=optimization_method
            )

            # Calculate additional metrics
            weights = optimization_results['weights']

            # Calculate risk contributions
            portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
            marginal_contrib = np.dot(covariance_matrix, weights)
            risk_contributions = weights * marginal_contrib / portfolio_variance

            # Calculate diversification ratio
            weighted_vol = np.sum(weights * np.sqrt(np.diag(covariance_matrix)))
            portfolio_vol = optimization_results['volatility']
            diversification_ratio = weighted_vol / portfolio_vol

            results = {
                **optimization_results,
                'risk_contributions': pd.Series(risk_contributions, index=assets),
                'diversification_ratio': diversification_ratio,
                'effective_assets': (risk_contributions > 0.01).sum(),
                'max_weight': weights.max(),
                'min_weight': weights.min(),
                'concentration': (weights ** 2).sum()  # Herfindahl index
            }

            logger.info(f"Portfolio optimization completed using {optimization_method}")
            logger.info(f"   Expected return: {results['expected_return']:.2%}")
            logger.info(f"   Volatility: {results['volatility']:.2%}")
            logger.info(f"   Sharpe ratio: {results['sharpe_ratio']:.2f}")
            logger.info(f"   Diversification ratio: {diversification_ratio:.2f}")

            return results

        except Exception as e:
            logger.error(f"Error optimizing portfolio: {str(e)}")
            return {'error': str(e)}

if __name__ == "__main__":
    # Example usage
    economic_system = EconomicModelingSystem()

    # Create sample data
    dates = pd.date_range(end=datetime.now(), periods=100, freq='H')
    price_data = pd.Series(50000 + np.random.randn(100).cumsum() * 500, index=dates)
    volume_data = pd.Series(1000 + np.random.rand(100) * 500, index=dates)

    # Sample order book
    order_book = {
        'bids': [(49900, 1.5), (49850, 2.0), (49800, 1.8)],
        'asks': [(50100, 1.2), (50150, 1.8), (50200, 2.1)]
    }

    # Sample trades
    recent_trades = [
        {'side': 'Buy', 'qty': 1.5, 'price': 50100},
        {'side': 'Sell', 'qty': 1.2, 'price': 50050}
    ]

    # Sample macro data
    macro_data = {
        'interest_rate_change': 0.0025,  # 25 bps
        'inflation_rate': 0.035,
        'risk_sentiment': 30,
        'dollar_index': 105,
        'market_volatility': 25
    }

    # Sample social metrics
    social_metrics = {
        'sentiment': 20,
        'volume_change': 150,
        'sentiment_change': 30,
        'search_interest': 0.8,
        'dominance': {'btc': 52}
    }

    # Analyze market
    analysis = economic_system.analyze_market_conditions(
        "BTCUSDT",
        order_book,
        recent_trades,
        price_data,
        volume_data,
        macro_data,
        social_metrics
    )

    print("Market Analysis Results:")
    print(f"Regime: {analysis['regime']}")
    print(f"Fear & Greed: {analysis['fear_greed']:.1f}")
    print(f"Liquidity Score: {analysis['liquidity']['liquidity_score']:.1f}")
    print(f"Overall Risk: {analysis['risk']['overall_risk_score']:.2f}")
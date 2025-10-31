#!/usr/bin/env python3
"""
Quantitative Trading Integration Layer

Integrates the sophisticated quantitative position sizing and economic modeling
systems with the existing trading engine to replace the primitive fixed $3.00
positioning with institutional-grade dynamic sizing.

This module serves as the bridge between:
- Quantitative position sizing models
- Economic modeling system
- Existing trading engine and risk management
"""

import asyncio
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass

# Import our quantitative models
from quantitative_position_sizing import (
    QuantitativePositionSizer,
    AssetMetrics,
    PortfolioMetrics,
    MarketPsychology,
    PositionSizeValidator
)
from economic_modeling_system import (
    EconomicModelingSystem,
    MarketRegime
)
from risk_management_system import RiskManager, RiskLevel

logger = logging.getLogger(__name__)

@dataclass
class TradingDecision:
    """Enhanced trading decision with quantitative backing"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float  # 0-1
    position_size_usd: float
    recommended_leverage: float
    expected_return: float
    risk_estimate: float
    sharpe_ratio: float
    market_regime: str
    fear_greed_index: float
    liquidity_score: float
    quantitative_reasoning: str
    risk_adjustments: List[str]

class QuantitativeTradingEngine:
    """
    Enhanced trading engine that integrates quantitative models

    Replaces primitive fixed position sizing with sophisticated
    quantitative analysis and economic modeling
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize quantitative trading engine

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Initialize quantitative components
        self.position_sizer = QuantitativePositionSizer(self.config)
        self.economic_model = EconomicModelingSystem(self.config)
        self.validator = PositionSizeValidator(num_simulations=5000)

        # Integration with existing risk management
        self.risk_manager = RiskManager()

        # Track performance metrics
        self.position_history = []
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'avg_position_size': 0.0
        }

        logger.info("Quantitative Trading Engine initialized with institutional-grade models")

    async def analyze_and_decide(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        portfolio_data: Dict[str, Any],
        signal_confidence: float = 0.7
    ) -> TradingDecision:
        """
        Analyze market conditions and make quantitative trading decision

        Args:
            symbol: Trading symbol
            market_data: Comprehensive market data
            portfolio_data: Current portfolio information
            signal_confidence: Base signal confidence from AI models

        Returns:
            Quantitative trading decision
        """
        try:
            logger.info(f"Performing quantitative analysis for {symbol}")

            # 1. Convert market data to quantitative metrics
            asset_metrics = self._create_asset_metrics(symbol, market_data)
            portfolio_metrics = self._create_portfolio_metrics(portfolio_data)
            market_psychology = self._create_market_psychology(market_data)

            # 2. Get historical data for analysis
            historical_data = self._extract_historical_data(market_data)

            # 3. Perform comprehensive market analysis
            market_analysis = await self._perform_market_analysis(
                symbol, market_data, historical_data
            )

            # 4. Calculate optimal position size
            optimal_position, sizing_details = self.position_sizer.calculate_optimal_position_size(
                symbol,
                asset_metrics,
                portfolio_metrics,
                market_psychology,
                historical_data,
                signal_confidence
            )

            # 5. Validate position sizing with Monte Carlo
            validation_results = self._validate_position_sizing(
                optimal_position, portfolio_metrics.total_value, asset_metrics
            )

            # 6. Integrate with existing risk management
            risk_assessment = self._integrate_risk_management(
                symbol, optimal_position, sizing_details, market_analysis
            )

            # 7. Make final decision
            decision = self._make_trading_decision(
                symbol, signal_confidence, optimal_position, sizing_details,
                market_analysis, risk_assessment, validation_results
            )

            # 8. Log the decision
            self._log_quantitative_decision(decision, sizing_details, market_analysis)

            return decision

        except Exception as e:
            logger.error(f"Error in quantitative analysis for {symbol}: {str(e)}")
            return self._create_fallback_decision(symbol, signal_confidence)

    def _create_asset_metrics(self, symbol: str, market_data: Dict) -> AssetMetrics:
        """Create asset metrics from market data"""
        price_data = market_data.get('price_data', {})
        technical_data = market_data.get('technical_data', {})
        market_metrics = market_data.get('market_metrics', {})

        return AssetMetrics(
            symbol=symbol,
            current_price=price_data.get('current_price', 0),
            volatility=market_metrics.get('volatility_30d', 0.8),
            atr_14d=technical_data.get('atr_14', price_data.get('current_price', 0) * 0.05),
            beta=market_metrics.get('beta', 1.0),
            sharpe_30d=market_metrics.get('sharpe_30d', 1.0),
            max_dd_30d=market_metrics.get('max_drawdown_30d', -0.15),
            correlation_btc=market_metrics.get('correlation_btc', 0.8),
            liquidity_score=market_metrics.get('liquidity_score', 70),
            funding_rate=market_metrics.get('funding_rate', 0.0001),
            open_interest_change=market_metrics.get('oi_change_24h', 0.05)
        )

    def _create_portfolio_metrics(self, portfolio_data: Dict) -> PortfolioMetrics:
        """Create portfolio metrics from portfolio data"""
        positions = portfolio_data.get('positions', {})
        total_value = portfolio_data.get('total_value', 0)
        available_cash = portfolio_data.get('available_balance', 0)

        # Create correlation matrix (simplified)
        symbols = list(positions.keys())
        if symbols:
            correlation_data = {s: {t: 0.5 if s != t else 1.0 for t in symbols} for s in symbols}
            correlation_matrix = pd.DataFrame(correlation_data)
        else:
            correlation_matrix = pd.DataFrame()

        return PortfolioMetrics(
            total_value=total_value,
            available_cash=available_cash,
            positions=positions,
            portfolio_volatility=portfolio_data.get('portfolio_volatility', 0.6),
            portfolio_beta=portfolio_data.get('portfolio_beta', 1.0),
            max_drawdown=portfolio_data.get('max_drawdown', -0.10),
            sharpe_ratio=portfolio_data.get('sharpe_ratio', 1.2),
            correlation_matrix=correlation_matrix
        )

    def _create_market_psychology(self, market_data: Dict) -> MarketPsychology:
        """Create market psychology indicators from market data"""
        sentiment_data = market_data.get('sentiment_data', {})
        fear_greed_data = market_data.get('fear_greed_data', {})

        return MarketPsychology(
            fear_greed_index=fear_greed_data.get('value', 50),
            fomo_intensity=sentiment_data.get('fomo_intensity', 30),
            sentiment_score=sentiment_data.get('sentiment_score', 0),
            social_volume=sentiment_data.get('social_volume', 1.0),
            volatility_premium=market_data.get('volatility_premium', 0.1)
        )

    def _extract_historical_data(self, market_data: Dict) -> Dict[str, pd.Series]:
        """Extract historical data from market data"""
        historical = market_data.get('historical_data', {})

        # Create pandas Series if not already
        price_data = historical.get('close', pd.Series())
        volume_data = historical.get('volume', pd.Series())
        volatility_data = historical.get('volatility', pd.Series())

        return {
            'price': price_data,
            'volume': volume_data,
            'volatility': volatility_data
        }

    async def _perform_market_analysis(
        self,
        symbol: str,
        market_data: Dict,
        historical_data: Dict[str, pd.Series]
    ) -> Dict[str, Any]:
        """Perform comprehensive market analysis"""
        try:
            # Simulate order book and recent trades (in real implementation, fetch from exchange)
            order_book = market_data.get('order_book', {})
            recent_trades = market_data.get('recent_trades', [])
            macro_data = market_data.get('macro_data', {})
            social_metrics = market_data.get('social_metrics', {})

            # Perform economic modeling analysis
            analysis = self.economic_model.analyze_market_conditions(
                symbol,
                order_book,
                recent_trades,
                historical_data['price'],
                historical_data['volume'],
                macro_data,
                social_metrics
            )

            return analysis

        except Exception as e:
            logger.error(f"Error in market analysis for {symbol}: {str(e)}")
            return {'error': str(e)}

    def _validate_position_sizing(
        self,
        position_size: float,
        portfolio_value: float,
        asset_metrics: AssetMetrics
    ) -> Dict[str, float]:
        """Validate position sizing using Monte Carlo simulation"""
        try:
            # Calculate position size as fraction of portfolio
            position_fraction = position_size / portfolio_value

            # Estimate expected return and volatility
            expected_return = asset_metrics.sharpe_30d * asset_metrics.volatility
            volatility = asset_metrics.volatility / np.sqrt(252)  # Daily volatility

            # Run Monte Carlo simulation
            validation_results = self.validator.validate_position_sizing(
                portfolio_value,
                position_fraction,
                expected_return / 252,  # Daily expected return
                volatility
            )

            return validation_results

        except Exception as e:
            logger.error(f"Error validating position sizing: {str(e)}")
            return {'error': str(e)}

    def _integrate_risk_management(
        self,
        symbol: str,
        position_size: float,
        sizing_details: Dict,
        market_analysis: Dict
    ) -> Dict[str, Any]:
        """Integrate with existing risk management system"""
        try:
            # Get risk parameters from sizing details
            risk_params = sizing_details.get('risk_params', {})
            recommended_leverage = sizing_details.get('recommended_leverage', 1.0)

            # Assess risk level
            risk_score = market_analysis.get('risk', {}).get('overall_risk_score', 0.5)
            if risk_score > 0.7:
                risk_level = RiskLevel.HIGH
            elif risk_score > 0.5:
                risk_level = RiskLevel.MODERATE
            elif risk_score > 0.3:
                risk_level = RiskLevel.LOW
            else:
                risk_level = RiskLevel.MINIMAL

            # Create risk assessment
            risk_assessment = {
                'risk_level': risk_level,
                'risk_score': risk_score,
                'max_leverage': risk_params.get('max_leverage', 5.0),
                'position_size_risk': position_size / sizing_details.get('portfolio_value', 10000),
                'leverage_risk': recommended_leverage / risk_params.get('max_leverage', 5.0),
                'regime_risk': 1.0 if market_analysis.get('regime') == 'NORMAL' else 0.5,
                'warnings': [],
                'adjustments': []
            }

            # Add warnings based on risk factors
            if recommended_leverage > risk_params.get('max_leverage', 5.0):
                risk_assessment['warnings'].append('Recommended leverage exceeds limits')
                risk_assessment['adjustments'].append('Reduce leverage to maximum allowed')

            if risk_score > 0.8:
                risk_assessment['warnings'].append('High market risk detected')
                risk_assessment['adjustments'].append('Consider reducing position size')

            if market_analysis.get('regime') == 'CRISIS':
                risk_assessment['warnings'].append('Market crisis regime detected')
                risk_assessment['adjustments'].append('Extreme caution advised')

            return risk_assessment

        except Exception as e:
            logger.error(f"Error in risk management integration: {str(e)}")
            return {'error': str(e)}

    def _make_trading_decision(
        self,
        symbol: str,
        signal_confidence: float,
        optimal_position: float,
        sizing_details: Dict,
        market_analysis: Dict,
        risk_assessment: Dict,
        validation_results: Dict
    ) -> TradingDecision:
        """Make final trading decision"""
        try:
            # Combine all confidence factors
            quantitative_confidence = min(0.9, signal_confidence)

            # Adjust confidence based on risk
            risk_score = risk_assessment.get('risk_score', 0.5)
            if risk_score > 0.7:
                quantitative_confidence *= 0.7
            elif risk_score > 0.5:
                quantitative_confidence *= 0.85

            # Adjust based on validation results
            probability_of_loss = validation_results.get('probability_of_loss', 0.3)
            if probability_of_loss > 0.4:
                quantitative_confidence *= 0.8

            # Determine action
            if quantitative_confidence > 0.6:
                action = "BUY"
            elif quantitative_confidence > 0.4:
                action = "HOLD"
            else:
                action = "SELL"

            # Apply risk adjustments
            adjusted_position = optimal_position
            if risk_assessment.get('adjustments'):
                if 'Reduce position size' in str(risk_assessment['adjustments']):
                    adjusted_position *= 0.7

            # Generate quantitative reasoning
            reasoning = self._generate_quantitative_reasoning(
                sizing_details, market_analysis, risk_assessment, validation_results
            )

            decision = TradingDecision(
                symbol=symbol,
                action=action,
                confidence=quantitative_confidence,
                position_size_usd=adjusted_position,
                recommended_leverage=sizing_details.get('recommended_leverage', 1.0),
                expected_return=sizing_details.get('expected_return', 0.1),
                risk_estimate=risk_assessment.get('risk_score', 0.5),
                sharpe_ratio=sizing_details.get('sharpe_ratio', 1.0),
                market_regime=market_analysis.get('regime', 'UNKNOWN'),
                fear_greed_index=market_analysis.get('fear_greed', 50),
                liquidity_score=market_analysis.get('liquidity', {}).get('liquidity_score', 50),
                quantitative_reasoning=reasoning,
                risk_adjustments=risk_assessment.get('adjustments', [])
            )

            return decision

        except Exception as e:
            logger.error(f"Error making trading decision: {str(e)}")
            return self._create_fallback_decision(symbol, signal_confidence)

    def _generate_quantitative_reasoning(
        self,
        sizing_details: Dict,
        market_analysis: Dict,
        risk_assessment: Dict,
        validation_results: Dict
    ) -> str:
        """Generate detailed quantitative reasoning for the decision"""
        reasoning_parts = []

        # Position sizing rationale
        kelly_size = sizing_details.get('kelly_size', 0)
        volatility_size = sizing_details.get('volatility_scaled', 0)
        base_position = sizing_details.get('base_position', 0)

        if kelly_size > 0:
            reasoning_parts.append(f"Kelly Criterion suggests ${kelly_size:.2f}")
        if volatility_size > 0:
            reasoning_parts.append(f"Volatility-adjusted size: ${volatility_size:.2f}")

        # Market regime rationale
        regime = market_analysis.get('regime', 'UNKNOWN')
        fear_greed = market_analysis.get('fear_greed', 50)
        reasoning_parts.append(f"Market regime: {regime}")
        reasoning_parts.append(f"Fear & Greed: {fear_greed:.1f}")

        # Risk assessment rationale
        risk_score = risk_assessment.get('risk_score', 0.5)
        reasoning_parts.append(f"Risk score: {risk_score:.2f}")

        # Validation rationale
        prob_loss = validation_results.get('probability_of_loss', 0.3)
        reasoning_parts.append(f"Loss probability: {prob_loss:.1%}")

        return "; ".join(reasoning_parts)

    def _create_fallback_decision(self, symbol: str, confidence: float) -> TradingDecision:
        """Create conservative fallback decision"""
        return TradingDecision(
            symbol=symbol,
            action="HOLD",
            confidence=confidence * 0.5,  # Reduce confidence
            position_size_usd=100.0,  # Conservative minimum
            recommended_leverage=1.0,  # No leverage
            expected_return=0.05,
            risk_estimate=0.5,
            sharpe_ratio=0.5,
            market_regime="UNKNOWN",
            fear_greed_index=50,
            liquidity_score=50,
            quantitative_reasoning="Fallback decision due to analysis error",
            risk_adjustments=["Conservative positioning due to error"]
        )

    def _log_quantitative_decision(
        self,
        decision: TradingDecision,
        sizing_details: Dict,
        market_analysis: Dict
    ):
        """Log the quantitative decision details"""
        logger.info(f"🎯 QUANTITATIVE TRADING DECISION: {decision.symbol}")
        logger.info(f"   Action: {decision.action}")
        logger.info(f"   Confidence: {decision.confidence:.1%}")
        logger.info(f"   Position Size: ${decision.position_size_usd:.2f}")
        logger.info(f"   Recommended Leverage: {decision.recommended_leverage:.1f}x")
        logger.info(f"   Market Regime: {decision.market_regime}")
        logger.info(f"   Fear & Greed: {decision.fear_greed_index:.1f}")
        logger.info(f"   Liquidity Score: {decision.liquidity_score:.1f}")
        logger.info(f"   Expected Return: {decision.expected_return:.1%}")
        logger.info(f"   Risk Estimate: {decision.risk_estimate:.1%}")
        logger.info(f"   Sharpe Ratio: {decision.sharpe_ratio:.2f}")
        logger.info(f"   Reasoning: {decision.quantitative_reasoning}")

        if decision.risk_adjustments:
            logger.info(f"   Risk Adjustments: {', '.join(decision.risk_adjustments)}")

    def update_performance_metrics(self, trade_result: Dict[str, Any]):
        """Update performance metrics after trade completion"""
        try:
            self.performance_metrics['total_trades'] += 1

            pnl = trade_result.get('pnl', 0)
            self.performance_metrics['total_pnl'] += pnl

            if pnl > 0:
                self.performance_metrics['winning_trades'] += 1

            # Update other metrics as needed
            # This would be expanded in a real implementation

            logger.info(f"Updated performance metrics: {self.performance_metrics}")

        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")

    def get_quantitative_summary(self) -> Dict[str, Any]:
        """Get summary of quantitative trading performance"""
        summary = {
            'performance_metrics': self.performance_metrics,
            'recent_decisions': self.position_history[-10:] if self.position_history else [],
            'model_status': {
                'position_sizer': 'active',
                'economic_model': 'active',
                'validator': 'active',
                'risk_integration': 'active'
            }
        }

        return summary

# Integration function to connect with existing trading engine
def create_quantitative_enhancement(
    existing_trading_engine,
    config: Optional[Dict] = None
) -> QuantitativeTradingEngine:
    """
    Create quantitative enhancement for existing trading engine

    Args:
        existing_trading_engine: Existing trading engine instance
        config: Configuration for quantitative models

    Returns:
        Enhanced quantitative trading engine
    """
    quantitative_engine = QuantitativeTradingEngine(config)

    # Store reference to existing engine for fallback
    quantitative_engine.existing_engine = existing_trading_engine

    logger.info("Quantitative enhancement created for existing trading engine")
    return quantitative_engine

if __name__ == "__main__":
    # Example usage
    quantitative_engine = QuantitativeTradingEngine()

    # Create sample market data
    market_data = {
        'price_data': {'current_price': 50000},
        'technical_data': {'atr_14': 1000},
        'market_metrics': {
            'volatility_30d': 0.8,
            'sharpe_30d': 1.2,
            'liquidity_score': 75
        },
        'sentiment_data': {
            'fomo_intensity': 40,
            'sentiment_score': 10
        },
        'fear_greed_data': {'value': 45},
        'historical_data': {
            'close': pd.Series([50000 + np.random.randn() * 1000 for _ in range(100)])
        }
    }

    portfolio_data = {
        'total_value': 10000,
        'available_balance': 8000,
        'positions': {'ETHUSDT': 2000}
    }

    # Run analysis (async in real usage)
    decision = asyncio.run(
        quantitative_engine.analyze_and_decide(
            "BTCUSDT",
            market_data,
            portfolio_data,
            signal_confidence=0.8
        )
    )

    print("Quantitative Trading Decision:")
    print(f"Action: {decision.action}")
    print(f"Position Size: ${decision.position_size_usd:.2f}")
    print(f"Leverage: {decision.recommended_leverage:.1f}x")
    print(f"Confidence: {decision.confidence:.1%}")
    print(f"Reasoning: {decision.quantitative_reasoning}")
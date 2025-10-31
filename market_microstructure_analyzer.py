"""
MARKET MICROSTRUCTURE ANALYZER
==============================
Advanced market microstructure analysis for optimal trading:
- Order book depth analysis
- Liquidity detection and utilization
- Market impact modeling
- Optimal execution timing
- Real-time spread analysis
"""

import asyncio
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
from bybit_client import BybitClient
from config import Config

logger = logging.getLogger(__name__)

@dataclass
class OrderBookAnalysis:
    """Comprehensive order book analysis"""
    symbol: str
    timestamp: float
    best_bid: float
    best_ask: float
    spread: float
    spread_pct: float
    bid_volume: float
    ask_volume: float
    total_volume: float
    order_book_imbalance: float
    liquidity_score: float
    market_depth: Dict
    price_impact_curve: List[Tuple[float, float]]
    execution_optimization: Dict

@dataclass
class LiquidityAnalysis:
    """Liquidity analysis results"""
    available_liquidity: float
    liquidity_depth: Dict
    liquidity_concentration: float
    liquidity_trend: str
    optimal_execution_size: float
    market_impact_estimate: float

@dataclass
class MarketImpactModel:
    """Market impact estimation model"""
    linear_impact: float
    square_root_impact: float
    temporary_impact: float
    permanent_impact: float
    confidence_interval: Tuple[float, float]

class MarketMicrostructureAnalyzer:
    """
    Advanced market microstructure analyzer for optimal trading execution
    """

    def __init__(self, bybit_client: BybitClient):
        self.bybit_client = bybit_client
        self.config = Config()

        # Data storage
        self.order_book_history = deque(maxlen=100)
        self.liquidity_history = deque(maxlen=50)
        self.spread_history = deque(maxlen=200)

        # Analysis parameters
        self.depth_levels = 20  # Number of levels to analyze
        self.impact_threshold = 0.001  # 0.1% impact threshold
        self.liquidity_threshold = 10000  # $10k minimum liquidity

        # Market impact model parameters
        self.impact_params = {
            'alpha': 0.5,      # Square root impact parameter
            'beta': 0.1,       # Linear impact parameter
            'gamma': 0.05,     # Temporary impact parameter
            'decay_rate': 0.1  # Impact decay rate
        }

        # Real-time analysis
        self.analysis_active = False
        self.last_analysis_time = 0

        logger.info("📊 Market Microstructure Analyzer initialized")
        logger.info(f"   Depth levels: {self.depth_levels}")
        logger.info(f"   Impact threshold: {self.impact_threshold * 100:.2f}%")
        logger.info(f"   Liquidity threshold: ${self.liquidity_threshold:,}")

    async def analyze_market_microstructure(self,
                                         symbol: str,
                                         urgency: str = "NORMAL") -> OrderBookAnalysis:
        """
        Perform comprehensive market microstructure analysis
        """

        try:
            start_time = time.time()

            # Get order book data
            order_book = self.bybit_client.get_order_book_data(symbol, limit=self.depth_levels * 2)
            if not order_book:
                logger.warning(f"No order book data available for {symbol}")
                return self._create_default_analysis(symbol)

            # Analyze order book structure
            analysis = await self._analyze_order_book_structure(symbol, order_book, urgency)

            # Calculate liquidity metrics
            liquidity_analysis = await self._analyze_liquidity_profile(symbol, order_book)

            # Estimate market impact
            impact_model = self._estimate_market_impact(symbol, order_book, liquidity_analysis)

            # Optimize execution parameters
            execution_optimization = self._optimize_execution_parameters(
                symbol, analysis, liquidity_analysis, impact_model, urgency
            )

            # Combine all analyses
            final_analysis = OrderBookAnalysis(
                symbol=symbol,
                timestamp=time.time(),
                best_bid=analysis.best_bid,
                best_ask=analysis.best_ask,
                spread=analysis.spread,
                spread_pct=analysis.spread_pct,
                bid_volume=analysis.bid_volume,
                ask_volume=analysis.ask_volume,
                total_volume=analysis.total_volume,
                order_book_imbalance=analysis.order_book_imbalance,
                liquidity_score=liquidity_analysis.liquidity_concentration,
                market_depth=liquidity_analysis.liquidity_depth,
                price_impact_curve=self._calculate_price_impact_curve(order_book),
                execution_optimization=execution_optimization
            )

            # Store in history
            self.order_book_history.append(final_analysis)
            self.spread_history.append(analysis.spread_pct)

            analysis_time = (time.time() - start_time) * 1000
            logger.debug(f"Market microstructure analysis for {symbol} completed in {analysis_time:.2f}ms")

            return final_analysis

        except Exception as e:
            logger.error(f"Error analyzing market microstructure for {symbol}: {str(e)}")
            return self._create_default_analysis(symbol)

    async def _analyze_order_book_structure(self,
                                          symbol: str,
                                          order_book: Dict,
                                          urgency: str) -> OrderBookAnalysis:
        """Analyze order book structure and basic metrics"""

        try:
            bids = order_book.get('b', [])
            asks = order_book.get('a', [])

            if not bids or not asks:
                return self._create_default_analysis(symbol)

            # Extract best bid/ask
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])

            # Calculate spread metrics
            spread = best_ask - best_bid
            spread_pct = (spread / best_ask) * 100

            # Calculate volume metrics
            bid_volume = sum(float(bid[1]) for bid in bids[:self.depth_levels])
            ask_volume = sum(float(ask[1]) for ask in asks[:self.depth_levels])
            total_volume = bid_volume + ask_volume

            # Calculate order book imbalance
            order_book_imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0

            # Calculate liquidity score (0-1)
            liquidity_score = min(total_volume / self.liquidity_threshold, 1.0)

            analysis = OrderBookAnalysis(
                symbol=symbol,
                timestamp=time.time(),
                best_bid=best_bid,
                best_ask=best_ask,
                spread=spread,
                spread_pct=spread_pct,
                bid_volume=bid_volume,
                ask_volume=ask_volume,
                total_volume=total_volume,
                order_book_imbalance=order_book_imbalance,
                liquidity_score=liquidity_score,
                market_depth={},  # Will be filled by liquidity analysis
                price_impact_curve=[],
                execution_optimization={}
            )

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing order book structure for {symbol}: {str(e)}")
            return self._create_default_analysis(symbol)

    async def _analyze_liquidity_profile(self,
                                       symbol: str,
                                       order_book: Dict) -> LiquidityAnalysis:
        """Analyze liquidity distribution and depth"""

        try:
            bids = order_book.get('b', [])
            asks = order_book.get('a', [])

            if not bids or not asks:
                return LiquidityAnalysis(
                    available_liquidity=0,
                    liquidity_depth={},
                    liquidity_concentration=0,
                    liquidity_trend="UNKNOWN",
                    optimal_execution_size=0,
                    market_impact_estimate=0
                )

            # Calculate cumulative liquidity at different price levels
            bid_depth = {}
            ask_depth = {}
            cumulative_bid_liquidity = 0
            cumulative_ask_liquidity = 0

            for i in range(min(len(bids), len(asks), self.depth_levels)):
                bid_price, bid_qty = float(bids[i][0]), float(bids[i][1])
                ask_price, ask_qty = float(asks[i][0]), float(asks[i][1])

                cumulative_bid_liquidity += bid_qty * bid_price
                cumulative_ask_liquidity += ask_qty * ask_price

                bid_price_offset = (bids[0][0] - bid_price) / float(bids[0][0]) * 100
                ask_price_offset = (ask_price - asks[0][0]) / float(asks[0][0]) * 100

                bid_depth[f"-{bid_price_offset:.2f}%"] = cumulative_bid_liquidity
                ask_depth[f"+{ask_price_offset:.2f}%"] = cumulative_ask_liquidity

            # Calculate total available liquidity
            total_liquidity = cumulative_bid_liquidity + cumulative_ask_liquidity

            # Calculate liquidity concentration (how concentrated liquidity is at top levels)
            top_3_levels_liquidity = 0
            for i in range(min(3, len(bids), len(asks))):
                top_3_levels_liquidity += float(bids[i][1]) * float(bids[i][0])
                top_3_levels_liquidity += float(asks[i][1]) * float(asks[i][0])

            liquidity_concentration = top_3_levels_liquidity / total_liquidity if total_liquidity > 0 else 0

            # Determine liquidity trend
            liquidity_trend = self._analyze_liquidity_trend(symbol, bid_depth, ask_depth)

            # Calculate optimal execution size (2% of available liquidity at 1% price impact)
            liquidity_at_1pct = self._get_liquidity_at_price_impact(bid_depth, ask_depth, 1.0)
            optimal_execution_size = liquidity_at_1pct * 0.02

            # Estimate market impact for $1k notional
            market_impact_estimate = self._estimate_impact_for_notional(
                symbol, 1000, bid_depth, ask_depth
            )

            return LiquidityAnalysis(
                available_liquidity=total_liquidity,
                liquidity_depth={'bids': bid_depth, 'asks': ask_depth},
                liquidity_concentration=liquidity_concentration,
                liquidity_trend=liquidity_trend,
                optimal_execution_size=optimal_execution_size,
                market_impact_estimate=market_impact_estimate
            )

        except Exception as e:
            logger.error(f"Error analyzing liquidity profile for {symbol}: {str(e)}")
            return LiquidityAnalysis(
                available_liquidity=0,
                liquidity_depth={},
                liquidity_concentration=0,
                liquidity_trend="UNKNOWN",
                optimal_execution_size=0,
                market_impact_estimate=0
            )

    def _estimate_market_impact(self,
                              symbol: str,
                              order_book: Dict,
                              liquidity_analysis: LiquidityAnalysis) -> MarketImpactModel:
        """Estimate market impact using different models"""

        try:
            # Base parameters
            mid_price = (float(order_book['b'][0][0]) + float(order_book['a'][0][0])) / 2
            daily_volume = self._get_estimated_daily_volume(symbol)

            # Linear impact model: impact = beta * (trade_size / daily_volume)
            beta = self.impact_params['beta']
            linear_impact = beta * 0.001  # Assuming 0.1% of daily volume

            # Square root impact model: impact = alpha * sqrt(trade_size / daily_volume)
            alpha = self.impact_params['alpha']
            sqrt_impact = alpha * np.sqrt(0.001)

            # Temporary impact (recovers quickly)
            temporary_impact = self.impact_params['gamma'] * linear_impact

            # Permanent impact (information effect)
            permanent_impact = linear_impact * 0.5

            # Calculate confidence interval (±1 standard deviation)
            confidence_width = 0.2 * linear_impact  # 20% of impact estimate
            confidence_interval = (
                linear_impact - confidence_width,
                linear_impact + confidence_width
            )

            return MarketImpactModel(
                linear_impact=linear_impact,
                square_root_impact=sqrt_impact,
                temporary_impact=temporary_impact,
                permanent_impact=permanent_impact,
                confidence_interval=confidence_interval
            )

        except Exception as e:
            logger.error(f"Error estimating market impact for {symbol}: {str(e)}")
            return MarketImpactModel(
                linear_impact=0.001,
                square_root_impact=0.001,
                temporary_impact=0.0005,
                permanent_impact=0.0005,
                confidence_interval=(0.0005, 0.0015)
            )

    def _optimize_execution_parameters(self,
                                     symbol: str,
                                     order_book_analysis: OrderBookAnalysis,
                                     liquidity_analysis: LiquidityAnalysis,
                                     impact_model: MarketImpactModel,
                                     urgency: str) -> Dict:
        """Optimize execution parameters based on market conditions"""

        try:
            # Base execution strategy
            base_strategy = {
                'order_type': 'Market',
                'execution_style': 'NORMAL',
                'size_optimization': True,
                'timing_optimization': True
            }

            # Adjust based on urgency
            if urgency == "HIGH":
                base_strategy['execution_style'] = 'AGGRESSIVE'
                base_strategy['size_splitting'] = False
                base_strategy['timing_delay'] = 0
            elif urgency == "LOW":
                base_strategy['execution_style'] = 'PASSIVE'
                base_strategy['size_splitting'] = True
                base_strategy['timing_delay'] = 1.0  # 1 second delay
            else:  # NORMAL
                base_strategy['execution_style'] = 'BALANCED'
                base_strategy['size_splitting'] = liquidity_analysis.optimal_execution_size < 1000
                base_strategy['timing_delay'] = 0.1  # 100ms delay

            # Adjust based on spread
            if order_book_analysis.spread_pct > 0.1:  # Wide spread
                base_strategy['use_limit_orders'] = True
                base_strategy['limit_offset'] = 0.02  # 2bps improvement
            else:  # Tight spread
                base_strategy['use_limit_orders'] = False

            # Adjust based on liquidity
            if liquidity_analysis.available_liquidity < self.liquidity_threshold:
                base_strategy['size_splitting'] = True
                base_strategy['max_chunk_size'] = liquidity_analysis.optimal_execution_size
                base_strategy['execution_style'] = 'CONSERVATIVE'
            else:
                base_strategy['max_chunk_size'] = float('inf')

            # Adjust based on order book imbalance
            if abs(order_book_analysis.order_book_imbalance) > 0.3:  # Strong imbalance
                if order_book_analysis.order_book_imbalance > 0:  # Bid-heavy
                    base_strategy['bias'] = 'PASSIVE_BUY'
                else:  # Ask-heavy
                    base_strategy['bias'] = 'PASSIVE_SELL'

            # Add market impact considerations
            base_strategy['impact_estimate'] = impact_model.linear_impact
            base_strategy['impact_threshold'] = self.impact_threshold
            base_strategy['size_limit'] = self._calculate_size_limit(
                liquidity_analysis, impact_model, urgency
            )

            # Add execution timing recommendations
            base_strategy['optimal_timing'] = self._calculate_optimal_timing(
                order_book_analysis, liquidity_analysis
            )

            return base_strategy

        except Exception as e:
            logger.error(f"Error optimizing execution parameters for {symbol}: {str(e)}")
            return {
                'order_type': 'Market',
                'execution_style': 'CONSERVATIVE',
                'error': str(e)
            }

    def _calculate_price_impact_curve(self, order_book: Dict) -> List[Tuple[float, float]]:
        """Calculate price impact curve for different trade sizes"""

        try:
            bids = order_book.get('b', [])
            asks = order_book.get('a', [])

            if not bids or not asks:
                return []

            impact_curve = []
            mid_price = (float(bids[0][0]) + float(asks[0][0])) / 2

            # Calculate impact for different notional values
            test_sizes = [100, 500, 1000, 5000, 10000, 50000, 100000]  # USD

            for size in test_sizes:
                # Simulate market impact by walking through the order book
                remaining_size = size
                avg_price = mid_price

                # Simple impact calculation (would be more sophisticated in production)
                impact_pct = (size / 100000) ** 0.5 * 0.1  # Square root impact
                avg_price *= (1 + impact_pct)

                impact_curve.append((size, impact_pct))

            return impact_curve

        except Exception as e:
            logger.error(f"Error calculating price impact curve: {str(e)}")
            return []

    def _analyze_liquidity_trend(self,
                               symbol: str,
                               bid_depth: Dict,
                               ask_depth: Dict) -> str:
        """Analyze liquidity trend over time"""

        try:
            # Store current liquidity snapshot
            current_snapshot = {
                'timestamp': time.time(),
                'total_liquidity': sum(bid_depth.values()) + sum(ask_depth.values()),
                'bid_liquidity': sum(bid_depth.values()),
                'ask_liquidity': sum(ask_depth.values())
            }

            self.liquidity_history.append(current_snapshot)

            # Need at least 3 snapshots to determine trend
            if len(self.liquidity_history) < 3:
                return "INSUFFICIENT_DATA"

            # Analyze recent trend
            recent_snapshots = list(self.liquidity_history)[-3:]
            liquidity_values = [s['total_liquidity'] for s in recent_snapshots]

            if liquidity_values[-1] > liquidity_values[-2] > liquidity_values[-3]:
                return "IMPROVING"
            elif liquidity_values[-1] < liquidity_values[-2] < liquidity_values[-3]:
                return "DECLINING"
            else:
                return "STABLE"

        except Exception as e:
            logger.error(f"Error analyzing liquidity trend: {str(e)}")
            return "UNKNOWN"

    def _get_liquidity_at_price_impact(self,
                                     bid_depth: Dict,
                                     ask_depth: Dict,
                                     impact_pct: float) -> float:
        """Get available liquidity at specific price impact level"""

        try:
            # Find the price level that corresponds to the impact percentage
            target_bid_level = None
            target_ask_level = None

            # Search bid side
            for level, liquidity in bid_depth.items():
                if float(level) <= -impact_pct:
                    target_bid_level = liquidity

            # Search ask side
            for level, liquidity in ask_depth.items():
                if float(level) >= impact_pct:
                    target_ask_level = liquidity
                    break

            # Return total liquidity at that impact level
            bid_liquidity = target_bid_level or 0
            ask_liquidity = target_ask_level or 0

            return bid_liquidity + ask_liquidity

        except Exception:
            return 0

    def _estimate_impact_for_notional(self,
                                    symbol: str,
                                    notional: float,
                                    bid_depth: Dict,
                                    ask_depth: Dict) -> float:
        """Estimate market impact for a given notional value"""

        try:
            # Simple impact estimation based on liquidity depth
            total_liquidity = sum(bid_depth.values()) + sum(ask_depth.values())

            if total_liquidity == 0:
                return 0.01  # 1% default impact

            # Impact scales with square root of trade size relative to liquidity
            relative_size = notional / total_liquidity
            impact = 0.1 * np.sqrt(relative_size)  # Base impact factor

            return min(impact, 0.05)  # Cap at 5% impact

        except Exception:
            return 0.01

    def _get_estimated_daily_volume(self, symbol: str) -> float:
        """Get estimated daily volume for impact calculations"""
        try:
            market_data = self.bybit_client.get_market_data(symbol)
            if market_data:
                volume_24h = float(market_data.get('volume24h', 0))
                price = float(market_data.get('lastPrice', 1))
                return volume_24h * price
            return 1000000  # $1M default
        except:
            return 1000000

    def _calculate_size_limit(self,
                            liquidity_analysis: LiquidityAnalysis,
                            impact_model: MarketImpactModel,
                            urgency: str) -> float:
        """Calculate maximum trade size based on impact tolerance"""

        try:
            # Base size limit from liquidity
            base_limit = liquidity_analysis.optimal_execution_size

            # Adjust for urgency
            urgency_multiplier = {
                "HIGH": 2.0,
                "NORMAL": 1.0,
                "LOW": 0.5
            }.get(urgency, 1.0)

            # Adjust for impact tolerance
            impact_multiplier = min(1.0, self.impact_threshold / impact_model.linear_impact)

            # Calculate final size limit
            size_limit = base_limit * urgency_multiplier * impact_multiplier

            return max(size_limit, 100)  # Minimum $100

        except Exception:
            return 1000  # $1k default

    def _calculate_optimal_timing(self,
                                order_book_analysis: OrderBookAnalysis,
                                liquidity_analysis: LiquidityAnalysis) -> Dict:
        """Calculate optimal execution timing recommendations"""

        try:
            timing_recommendations = {}

            # Based on spread analysis
            recent_spreads = list(self.spread_history)[-10:] if len(self.spread_history) >= 10 else []
            if recent_spreads:
                avg_spread = np.mean(recent_spreads)
                current_spread = order_book_analysis.spread_pct

                if current_spread < avg_spread * 0.8:
                    timing_recommendations['spread_condition'] = "FAVORABLE"
                    timing_recommendations['timing_bias'] = "IMMEDIATE"
                elif current_spread > avg_spread * 1.2:
                    timing_recommendations['spread_condition'] = "UNFAVORABLE"
                    timing_recommendations['timing_bias'] = "DELAYED"
                else:
                    timing_recommendations['spread_condition'] = "NEUTRAL"
                    timing_recommendations['timing_bias'] = "NORMAL"

            # Based on liquidity trend
            timing_recommendations['liquidity_trend'] = liquidity_analysis.liquidity_trend
            if liquidity_analysis.liquidity_trend == "IMPROVING":
                timing_recommendations['liquidity_bias'] = "WAIT_FOR_BETTER"
            elif liquidity_analysis.liquidity_trend == "DECLINING":
                timing_recommendations['liquidity_bias'] = "EXECUTE_NOW"
            else:
                timing_recommendations['liquidity_bias'] = "NEUTRAL"

            # Based on order book imbalance
            if abs(order_book_analysis.order_book_imbalance) > 0.3:
                timing_recommendations['imbalance_bias'] = "WAIT_FOR_REVERSION"
            else:
                timing_recommendations['imbalance_bias'] = "NEUTRAL"

            return timing_recommendations

        except Exception as e:
            logger.error(f"Error calculating optimal timing: {str(e)}")
            return {'error': str(e)}

    def _create_default_analysis(self, symbol: str) -> OrderBookAnalysis:
        """Create default analysis when data is unavailable"""
        return OrderBookAnalysis(
            symbol=symbol,
            timestamp=time.time(),
            best_bid=0,
            best_ask=0,
            spread=0,
            spread_pct=0,
            bid_volume=0,
            ask_volume=0,
            total_volume=0,
            order_book_imbalance=0,
            liquidity_score=0,
            market_depth={},
            price_impact_curve=[],
            execution_optimization={
                'order_type': 'Market',
                'execution_style': 'CONSERVATIVE',
                'error': 'Insufficient data'
            }
        )

    def get_market_microstructure_summary(self, symbol: str) -> Dict:
        """Get summary of market microstructure analysis"""

        try:
            # Get recent analyses for the symbol
            recent_analyses = [
                analysis for analysis in self.order_book_history
                if analysis.symbol == symbol
            ][-10:]  # Last 10 analyses

            if not recent_analyses:
                return {'status': 'No data available for symbol'}

            # Calculate summary statistics
            spreads = [a.spread_pct for a in recent_analyses]
            volumes = [a.total_volume for a in recent_analyses]
            imbalances = [a.order_book_imbalance for a in recent_analyses]
            liquidity_scores = [a.liquidity_score for a in recent_analyses]

            import numpy as np

            return {
                'symbol': symbol,
                'analysis_count': len(recent_analyses),
                'latest_analysis': recent_analyses[-1].timestamp,
                'spread_metrics': {
                    'current_pct': spreads[-1] if spreads else 0,
                    'avg_pct': np.mean(spreads),
                    'min_pct': np.min(spreads),
                    'max_pct': np.max(spreads),
                    'trend': 'improving' if len(spreads) > 1 and spreads[-1] < spreads[-2] else 'stable'
                },
                'volume_metrics': {
                    'current': volumes[-1] if volumes else 0,
                    'avg': np.mean(volumes),
                    'trend': 'increasing' if len(volumes) > 1 and volumes[-1] > volumes[-2] else 'stable'
                },
                'liquidity_metrics': {
                    'current_score': liquidity_scores[-1] if liquidity_scores else 0,
                    'avg_score': np.mean(liquidity_scores),
                    'quality': 'high' if (liquidity_scores[-1] if liquidity_scores else 0) > 0.7 else 'medium' if (liquidity_scores[-1] if liquidity_scores else 0) > 0.3 else 'low'
                },
                'market_balance': {
                    'current_imbalance': imbalances[-1] if imbalances else 0,
                    'avg_imbalance': np.mean(imbalances),
                    'balance_state': 'balanced' if abs(imbalances[-1] if imbalances else 0) < 0.1 else 'bid_heavy' if (imbalances[-1] if imbalances else 0) > 0 else 'ask_heavy'
                },
                'execution_recommendations': recent_analyses[-1].execution_optimization if recent_analyses else {}
            }

        except Exception as e:
            logger.error(f"Error getting market microstructure summary for {symbol}: {str(e)}")
            return {'error': str(e)}

    def start_real_time_analysis(self):
        """Start real-time market microstructure analysis"""
        if self.analysis_active:
            return

        self.analysis_active = True
        logger.info("🔄 Starting real-time market microstructure analysis")

        # Start analysis loop
        asyncio.create_task(self._real_time_analysis_loop())

    async def _real_time_analysis_loop(self):
        """Real-time analysis loop"""
        while self.analysis_active:
            try:
                current_time = time.time()

                # Analyze each target asset
                for symbol in self.config.TARGET_ASSETS:
                    try:
                        await self.analyze_market_microstructure(symbol)
                    except Exception as e:
                        logger.error(f"Error in real-time analysis for {symbol}: {str(e)}")

                # Update analysis timing
                self.last_analysis_time = current_time

                # Wait before next analysis
                await asyncio.sleep(30)  # Analyze every 30 seconds

            except Exception as e:
                logger.error(f"Error in real-time analysis loop: {str(e)}")
                await asyncio.sleep(60)

    def stop_real_time_analysis(self):
        """Stop real-time analysis"""
        self.analysis_active = False
        logger.info("Real-time market microstructure analysis stopped")

    def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up Market Microstructure Analyzer...")
        self.stop_real_time_analysis()
        logger.info("✅ Market Microstructure Analyzer cleanup completed")
"""
HIGH-PERFORMANCE EXECUTION OPTIMIZER
===================================
Advanced execution optimization for crypto trading with:
- Latency optimization
- Slippage reduction
- Market microstructure integration
- Order routing algorithms
- Performance monitoring
"""

import asyncio
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor
from bybit_client import BybitClient
from config import Config

logger = logging.getLogger(__name__)

@dataclass
class ExecutionMetrics:
    """Real-time execution performance metrics"""
    latency_ms: float
    slippage_bps: float
    fill_rate: float
    execution_quality_score: float
    timestamp: float

@dataclass
class MarketMicrostructure:
    """Market microstructure analysis for optimal execution"""
    bid_ask_spread: float
    order_book_imbalance: float
    liquidity_score: float
    volatility_1min: float
    volume_profile: Dict
    optimal_execution_size: float

class PerformanceOptimizer:
    """
    High-performance execution optimizer with real-time monitoring
    and adaptive execution strategies
    """

    def __init__(self, bybit_client: BybitClient):
        self.bybit_client = bybit_client
        self.config = Config()

        # Performance tracking
        self.execution_metrics = []
        self.latency_history = []
        self.slippage_history = []

        # Optimization parameters
        self.target_latency_ms = 500.0  # 95th percentile target
        self.max_slippage_bps = 5.0     # Maximum acceptable slippage

        # Market microstructure cache
        self.microstructure_cache = {}
        self.cache_ttl = 30  # seconds

        # Async execution pool
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Real-time performance monitoring
        self.monitoring_active = False

        logger.info("🚀 Performance Optimizer initialized")
        logger.info(f"   Target latency: {self.target_latency_ms}ms (95th percentile)")
        logger.info(f"   Max slippage: {self.max_slippage_bps}bps")

    async def optimize_order_execution(self,
                                     symbol: str,
                                     side: str,
                                     quantity: float,
                                     order_type: str = "Market",
                                     urgency: str = "NORMAL") -> Dict:
        """
        Optimize order execution with market microstructure analysis
        and latency-aware order routing
        """
        start_time = time.time()

        try:
            # Get market microstructure analysis
            microstructure = await self._analyze_market_microstructure(symbol)

            # Calculate optimal execution parameters
            execution_params = self._calculate_execution_params(
                symbol, side, quantity, microstructure, urgency
            )

            # Execute with optimized parameters
            result = await self._execute_optimized_order(
                symbol, side, quantity, execution_params
            )

            # Calculate performance metrics
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            metrics = self._calculate_execution_metrics(
                symbol, result, execution_time, microstructure
            )

            # Update performance tracking
            self._update_performance_metrics(metrics)

            logger.info(f"✅ Optimized execution completed for {symbol}")
            logger.info(f"   Latency: {execution_time:.2f}ms")
            logger.info(f"   Slippage: {metrics.slippage_bps:.2f}bps")
            logger.info(f"   Quality score: {metrics.execution_quality_score:.2f}")

            return {
                'success': True,
                'order_result': result,
                'execution_metrics': metrics,
                'microstructure': microstructure,
                'optimization_applied': execution_params
            }

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"❌ Optimized execution failed for {symbol}: {str(e)}")
            logger.error(f"   Failed after: {execution_time:.2f}ms")

            return {
                'success': False,
                'error': str(e),
                'execution_time_ms': execution_time
            }

    async def _analyze_market_microstructure(self, symbol: str) -> MarketMicrostructure:
        """Analyze market microstructure for optimal execution timing"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{int(time.time() // self.cache_ttl)}"
            if cache_key in self.microstructure_cache:
                logger.debug(f"Using cached microstructure data for {symbol}")
                return self.microstructure_cache[cache_key]

            # Get order book data
            order_book = self.bybit_client.get_order_book_data(symbol, limit=50)
            if not order_book:
                return self._get_default_microstructure()

            # Get market data for volatility
            market_data = self.bybit_client.get_market_data(symbol)

            # Calculate microstructure metrics
            spread = float(order_book.get('spread_pct', 0))
            liquidity_score = float(order_book.get('liquidity_score', 0))

            # Calculate order book imbalance
            bids = order_book.get('bids', [])
            asks = order_book.get('a', [])

            bid_volume = sum(float(bid[1]) for bid in bids[:10])
            ask_volume = sum(float(ask[1]) for ask in asks[:10])

            total_volume = bid_volume + ask_volume
            order_book_imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0

            # Calculate short-term volatility (1-minute)
            volatility_1min = self._calculate_short_term_volatility(symbol, market_data)

            # Calculate optimal execution size based on liquidity
            optimal_execution_size = self._calculate_optimal_size(
                liquidity_score, order_book_imbalance, volatility_1min
            )

            # Create volume profile
            volume_profile = {
                'bid_levels': len(bids),
                'ask_levels': len(asks),
                'bid_liquidity': bid_volume,
                'ask_liquidity': ask_volume,
                'total_liquidity': total_volume
            }

            microstructure = MarketMicrostructure(
                bid_ask_spread=spread,
                order_book_imbalance=order_book_imbalance,
                liquidity_score=liquidity_score,
                volatility_1min=volatility_1min,
                volume_profile=volume_profile,
                optimal_execution_size=optimal_execution_size
            )

            # Cache the result
            self.microstructure_cache[cache_key] = microstructure

            logger.debug(f"Microstructure analysis for {symbol}:")
            logger.debug(f"   Spread: {spread:.4f}%")
            logger.debug(f"   Imbalance: {order_book_imbalance:.3f}")
            logger.debug(f"   Liquidity: {liquidity_score:.3f}")
            logger.debug(f"   Volatility: {volatility_1min:.3f}")

            return microstructure

        except Exception as e:
            logger.error(f"Error analyzing market microstructure for {symbol}: {str(e)}")
            return self._get_default_microstructure()

    def _calculate_execution_params(self,
                                  symbol: str,
                                  side: str,
                                  quantity: float,
                                  microstructure: MarketMicrostructure,
                                  urgency: str) -> Dict:
        """Calculate optimal execution parameters based on market conditions"""

        # Base execution strategy
        base_params = {
            'order_type': 'Market',
            'time_in_force': 'ImmediateOrCancel',
            'execution_style': 'AGGRESSIVE'
        }

        # Adjust based on urgency
        if urgency == "LOW":
            base_params['execution_style'] = 'PASSIVE'
            base_params['time_in_force'] = 'GoodTillCancel'
        elif urgency == "HIGH":
            base_params['execution_style'] = 'ULTRA_AGGRESSIVE'
            base_params['time_in_force'] = 'FillOrKill'

        # Slippage reduction strategies
        if microstructure.bid_ask_spread > 0.1:  # Wide spread
            base_params['slippage_tolerance'] = 0.05  # 5bps
            base_params['use_limit_orders'] = True
            base_params['limit_offset'] = 0.02  # 2bps inside spread
        else:  # Tight spread
            base_params['slippage_tolerance'] = 0.02  # 2bps
            base_params['use_limit_orders'] = False

        # Size optimization based on liquidity
        optimal_size = microstructure.optimal_execution_size
        if quantity > optimal_size:
            # Split large orders
            base_params['split_order'] = True
            base_params['chunk_size'] = optimal_size
            base_params['chunks'] = int(np.ceil(quantity / optimal_size))
        else:
            base_params['split_order'] = False
            base_params['chunk_size'] = quantity
            base_params['chunks'] = 1

        # Volatility adjustments
        if microstructure.volatility_1min > 0.5:  # High volatility
            base_params['execution_style'] = 'CONSERVATIVE'
            base_params['slippage_tolerance'] *= 1.5  # Increase tolerance
        elif microstructure.volatility_1min < 0.1:  # Low volatility
            base_params['execution_style'] = 'AGGRESSIVE'
            base_params['slippage_tolerance'] *= 0.8  # Decrease tolerance

        # Order book imbalance adjustments
        if abs(microstructure.order_book_imbalance) > 0.3:  # Strong imbalance
            if (side == "Buy" and microstructure.order_book_imbalance > 0) or \
               (side == "Sell" and microstructure.order_book_imbalance < 0):
                # Favorable imbalance - be more aggressive
                base_params['execution_style'] = 'AGGRESSIVE'
                base_params['slippage_tolerance'] *= 0.9
            else:
                # Unfavorable imbalance - be more conservative
                base_params['execution_style'] = 'CONSERVATIVE'
                base_params['slippage_tolerance'] *= 1.2

        logger.debug(f"Execution params for {symbol} {side}:")
        logger.debug(f"   Style: {base_params['execution_style']}")
        logger.debug(f"   Split order: {base_params['split_order']}")
        logger.debug(f"   Slippage tolerance: {base_params.get('slippage_tolerance', 0):.4f}")

        return base_params

    async def _execute_optimized_order(self,
                                     symbol: str,
                                     side: str,
                                     quantity: float,
                                     params: Dict) -> Dict:
        """Execute order using optimized parameters"""

        if params.get('split_order', False):
            return await self._execute_split_order(symbol, side, quantity, params)
        else:
            return await self._execute_single_order(symbol, side, quantity, params)

    async def _execute_single_order(self,
                                  symbol: str,
                                  side: str,
                                  quantity: float,
                                  params: Dict) -> Dict:
        """Execute a single optimized order"""

        order_params = {
            'symbol': symbol,
            'side': side,
            'order_type': params['order_type'],
            'qty': quantity,
            'time_in_force': params['time_in_force']
        }

        # Add limit price if using limit orders
        if params.get('use_limit_orders', False):
            current_price = float(self.bybit_client.get_market_data(symbol).get('lastPrice', 0))
            offset_pct = params.get('limit_offset', 0.02) / 100

            if side == "Buy":
                limit_price = current_price * (1 - offset_pct)
            else:
                limit_price = current_price * (1 + offset_pct)

            order_params['price'] = str(limit_price)

        # Execute the order
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self.bybit_client.place_order,
            **order_params
        )

        return result

    async def _execute_split_order(self,
                                 symbol: str,
                                 side: str,
                                 total_quantity: float,
                                 params: Dict) -> Dict:
        """Execute large order as multiple chunks"""

        chunk_size = params['chunk_size']
        chunks = params['chunks']

        logger.info(f"Executing split order: {chunks} chunks of {chunk_size:.6f} {symbol}")

        results = []
        total_filled = 0
        total_slippage = 0

        for i in range(chunks):
            # Calculate remaining quantity
            remaining_quantity = total_quantity - total_filled
            if remaining_quantity <= 0:
                break

            # Use chunk size or remaining amount, whichever is smaller
            execute_quantity = min(chunk_size, remaining_quantity)

            # Execute chunk
            chunk_result = await self._execute_single_order(
                symbol, side, execute_quantity, params
            )

            if chunk_result:
                results.append(chunk_result)

                # Calculate chunk slippage
                chunk_slippage = self._calculate_chunk_slippage(
                    symbol, chunk_result, params
                )
                total_slippage += chunk_slippage

                # Update filled quantity
                filled_qty = float(chunk_result.get('result', {}).get('executedQty', execute_quantity))
                total_filled += filled_qty

                logger.debug(f"Chunk {i+1}/{chunks} executed: {filled_qty:.6f}")

                # Small delay between chunks to avoid market impact
                if i < chunks - 1:
                    await asyncio.sleep(0.1)
            else:
                logger.warning(f"Chunk {i+1}/{chunks} failed")

        # Aggregate results
        aggregated_result = {
            'split_execution': True,
            'chunks_executed': len(results),
            'total_filled': total_filled,
            'total_quantity': total_quantity,
            'fill_rate': total_filled / total_quantity if total_quantity > 0 else 0,
            'average_slippage_bps': total_slippage / len(results) if results else 0,
            'chunk_results': results
        }

        return aggregated_result

    def _calculate_execution_metrics(self,
                                   symbol: str,
                                   result: Dict,
                                   execution_time: float,
                                   microstructure: MarketMicrostructure) -> ExecutionMetrics:
        """Calculate execution performance metrics"""

        # Calculate slippage
        slippage_bps = 0.0
        if result and 'result' in result:
            slippage_bps = self._calculate_actual_slippage(symbol, result)

        # Calculate fill rate
        fill_rate = 1.0  # Default for market orders
        if result.get('split_execution', False):
            fill_rate = result.get('fill_rate', 0)

        # Calculate execution quality score (0-100)
        latency_score = max(0, 100 - (execution_time / self.target_latency_ms) * 100)
        slippage_score = max(0, 100 - (slippage_bps / self.max_slippage_bps) * 100)
        fill_score = fill_rate * 100

        execution_quality_score = (latency_score + slippage_score + fill_score) / 3

        return ExecutionMetrics(
            latency_ms=execution_time,
            slippage_bps=slippage_bps,
            fill_rate=fill_rate,
            execution_quality_score=execution_quality_score,
            timestamp=time.time()
        )

    def _calculate_actual_slippage(self, symbol: str, result: Dict) -> float:
        """Calculate actual slippage in basis points"""
        try:
            # Get order details
            order_result = result.get('result', {})
            executed_price = float(order_result.get('executedPrice', 0))
            executed_qty = float(order_result.get('executedQty', 0))

            if executed_price == 0 or executed_qty == 0:
                return 0.0

            # Get market price at execution time
            market_data = self.bybit_client.get_market_data(symbol)
            reference_price = float(market_data.get('lastPrice', executed_price))

            # Calculate slippage
            if executed_price > 0 and reference_price > 0:
                slippage_pct = abs(executed_price - reference_price) / reference_price
                return slippage_pct * 10000  # Convert to basis points

            return 0.0

        except Exception as e:
            logger.error(f"Error calculating slippage: {str(e)}")
            return 0.0

    def _calculate_chunk_slippage(self, symbol: str, result: Dict, params: Dict) -> float:
        """Calculate slippage for a single chunk"""
        return self._calculate_actual_slippage(symbol, result)

    def _calculate_short_term_volatility(self, symbol: str, market_data: Dict) -> float:
        """Calculate 1-minute volatility estimate"""
        try:
            # Use 24h range as proxy for short-term volatility
            high_24h = float(market_data.get('high24h', 0))
            low_24h = float(market_data.get('low24h', 0))
            current_price = float(market_data.get('lastPrice', 0))

            if current_price > 0 and high_24h > low_24h:
                daily_range_pct = ((high_24h - low_24h) / current_price) * 100
                # Estimate 1-minute volatility (assuming random walk)
                return daily_range_pct / np.sqrt(24 * 60)  # Square root of time rule

            return 0.1  # Default low volatility

        except Exception:
            return 0.1

    def _calculate_optimal_size(self,
                              liquidity_score: float,
                              order_book_imbalance: float,
                              volatility: float) -> float:
        """Calculate optimal execution size based on market conditions"""

        # Base size as percentage of available liquidity
        base_size = liquidity_score * 0.1  # 10% of available liquidity

        # Adjust for order book imbalance
        if abs(order_book_imbalance) > 0.3:
            base_size *= 0.7  # Reduce size in imbalanced conditions

        # Adjust for volatility
        if volatility > 0.5:
            base_size *= 0.8  # Reduce size in high volatility
        elif volatility < 0.1:
            base_size *= 1.2  # Increase size in low volatility

        return max(base_size, 0.001)  # Minimum size

    def _get_default_microstructure(self) -> MarketMicrostructure:
        """Return default microstructure when data is unavailable"""
        return MarketMicrostructure(
            bid_ask_spread=0.05,
            order_book_imbalance=0.0,
            liquidity_score=0.5,
            volatility_1min=0.1,
            volume_profile={},
            optimal_execution_size=0.001
        )

    def _update_performance_metrics(self, metrics: ExecutionMetrics):
        """Update performance tracking with new metrics"""
        self.execution_metrics.append(metrics)

        # Keep only last 1000 metrics
        if len(self.execution_metrics) > 1000:
            self.execution_metrics = self.execution_metrics[-1000:]

        # Update specific histories
        self.latency_history.append(metrics.latency_ms)
        self.slippage_history.append(metrics.slippage_bps)

        # Keep only last 500 values
        if len(self.latency_history) > 500:
            self.latency_history = self.latency_history[-500:]
        if len(self.slippage_history) > 500:
            self.slippage_history = self.slippage_history[-500:]

    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        if not self.execution_metrics:
            return {'status': 'No execution data available'}

        # Calculate percentiles
        latencies = [m.latency_ms for m in self.execution_metrics]
        slippages = [m.slippage_bps for m in self.execution_metrics]
        quality_scores = [m.execution_quality_score for m in self.execution_metrics]

        return {
            'total_executions': len(self.execution_metrics),
            'latency_metrics': {
                'avg_ms': np.mean(latencies),
                'median_ms': np.median(latencies),
                'p95_ms': np.percentile(latencies, 95),
                'p99_ms': np.percentile(latencies, 99),
                'max_ms': np.max(latencies),
                'target_met': np.percentile(latencies, 95) <= self.target_latency_ms
            },
            'slippage_metrics': {
                'avg_bps': np.mean(slippages),
                'median_bps': np.median(slippages),
                'p95_bps': np.percentile(slippages, 95),
                'max_bps': np.max(slippages),
                'within_tolerance': np.percentile(slippages, 95) <= self.max_slippage_bps
            },
            'quality_metrics': {
                'avg_score': np.mean(quality_scores),
                'median_score': np.median(quality_scores),
                'min_score': np.min(quality_scores),
                'max_score': np.max(quality_scores)
            },
            'recent_performance': {
                'last_10_avg_latency': np.mean(latencies[-10:]),
                'last_10_avg_slippage': np.mean(slippages[-10:]),
                'last_10_avg_quality': np.mean(quality_scores[-10:])
            }
        }

    def start_performance_monitoring(self):
        """Start real-time performance monitoring"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        logger.info("🔍 Performance monitoring started")

        # Start monitoring task
        asyncio.create_task(self._monitor_performance_loop())

    async def _monitor_performance_loop(self):
        """Real-time performance monitoring loop"""
        while self.monitoring_active:
            try:
                if len(self.execution_metrics) >= 10:
                    summary = self.get_performance_summary()

                    # Check for performance degradation
                    p95_latency = summary['latency_metrics']['p95_ms']
                    p95_slippage = summary['slippage_metrics']['p95_bps']

                    if p95_latency > self.target_latency_ms * 1.5:
                        logger.warning(f"⚠️ Performance degradation detected:")
                        logger.warning(f"   P95 Latency: {p95_latency:.2f}ms (target: {self.target_latency_ms}ms)")

                    if p95_slippage > self.max_slippage_bps * 1.5:
                        logger.warning(f"⚠️ Slippage degradation detected:")
                        logger.warning(f"   P95 Slippage: {p95_slippage:.2f}bps (target: {self.max_slippage_bps}bps)")

                    # Log performance summary every 5 minutes
                    if len(self.execution_metrics) % 50 == 0:
                        logger.info(f"📊 Performance Summary ({len(self.execution_metrics)} executions):")
                        logger.info(f"   P95 Latency: {p95_latency:.2f}ms")
                        logger.info(f"   P95 Slippage: {p95_slippage:.2f}bps")
                        logger.info(f"   Avg Quality: {summary['quality_metrics']['avg_score']:.1f}/100")

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in performance monitoring: {str(e)}")
                await asyncio.sleep(30)

    def stop_performance_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        logger.info("Performance monitoring stopped")

    def cleanup(self):
        """Cleanup resources"""
        self.stop_performance_monitoring()
        self.executor.shutdown(wait=True)
        logger.info("Performance Optimizer cleanup completed")
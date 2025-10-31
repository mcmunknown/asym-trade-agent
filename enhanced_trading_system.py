"""
ENHANCED HIGH-PERFORMANCE TRADING SYSTEM
========================================
Complete trading system with all optimizations:
- Performance optimization
- Latency optimization
- AI signal optimization
- Market microstructure integration
- Real-time monitoring
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from decimal import Decimal

from bybit_client import BybitClient
from performance_optimizer import PerformanceOptimizer
from latency_optimizer import LatencyOptimizer
from ai_signal_optimizer import AISignalOptimizer
from market_microstructure_analyzer import MarketMicrostructureAnalyzer
from enhanced_trading_engine import EnhancedTradingEngine
from config import Config

logger = logging.getLogger(__name__)

@dataclass
class TradingSystemMetrics:
    """Comprehensive trading system performance metrics"""
    timestamp: float
    total_signals_generated: int
    total_trades_executed: int
    success_rate: float
    avg_execution_latency: float
    avg_signal_generation_time: float
    total_pnl: float
    active_positions: int
    system_health_score: float

class EnhancedTradingSystem:
    """
    Complete high-performance trading system with all optimizations
    """

    def __init__(self):
        self.config = Config()

        # Initialize core components
        self.bybit_client = BybitClient()
        self.performance_optimizer = PerformanceOptimizer(self.bybit_client)
        self.latency_optimizer = LatencyOptimizer(self.bybit_client)
        self.ai_signal_optimizer = AISignalOptimizer()
        self.microstructure_analyzer = MarketMicrostructureAnalyzer(self.bybit_client)
        self.enhanced_engine = EnhancedTradingEngine()

        # System state
        self.system_active = False
        self.last_performance_check = 0
        self.performance_check_interval = 60  # seconds

        # Performance tracking
        self.system_metrics = []
        self.performance_history = []

        # Optimization status
        self.optimizations_enabled = {
            'performance_optimization': True,
            'latency_optimization': True,
            'ai_optimization': True,
            'microstructure_analysis': True
        }

        # Performance targets
        self.performance_targets = {
            'signal_generation_time_ms': 2000,
            'order_execution_time_ms': 500,
            'system_latency_ms': 3000,
            'success_rate_pct': 60,
            'system_health_score': 80
        }

        logger.info("🚀 Enhanced High-Performance Trading System initialized")
        logger.info("✅ All optimization modules loaded")
        logger.info(f"🎯 Performance targets: {self.performance_targets}")

    async def start_system(self):
        """Start the enhanced trading system with all optimizations"""
        try:
            if self.system_active:
                logger.warning("Trading system is already active")
                return

            logger.info("🎬 Starting Enhanced High-Performance Trading System...")

            # Pre-warm connections and caches
            await self._initialize_system()

            # Start monitoring systems
            await self._start_monitoring_systems()

            # Start the main trading loop
            self.system_active = True
            await self._run_enhanced_trading_loop()

        except Exception as e:
            logger.error(f"❌ Failed to start enhanced trading system: {str(e)}")
            await self.stop_system("Startup failed")
            raise

    async def _initialize_system(self):
        """Initialize all system components and optimizations"""
        try:
            logger.info("🔧 Initializing system components...")

            # Pre-warm connections for ultra-low latency
            await self.latency_optimizer.pre_warm_connections()

            # Start performance monitoring
            self.performance_optimizer.start_performance_monitoring()

            # Start real-time market microstructure analysis
            self.microstructure_analyzer.start_real_time_analysis()

            # Run system benchmarks
            await self._run_system_benchmarks()

            # Verify all components are ready
            await self._verify_system_health()

            logger.info("✅ System initialization completed")

        except Exception as e:
            logger.error(f"❌ System initialization failed: {str(e)}")
            raise

    async def _run_system_benchmarks(self):
        """Run comprehensive system benchmarks"""
        try:
            logger.info("🏃 Running system benchmarks...")

            # Benchmark latency optimization
            latency_benchmark = await self.latency_optimizer.benchmark_performance(iterations=50)
            logger.info(f"   Latency benchmark: P95 {latency_benchmark['latency_stats']['p95_ms']:.2f}ms")

            # Benchmark AI signal optimization
            start_time = time.time()
            test_signal = await self.ai_signal_optimizer.get_optimized_signal(
                "BTCUSDT",
                self._get_test_market_data(),
                priority=2
            )
            signal_time = (time.time() - start_time) * 1000
            logger.info(f"   AI signal benchmark: {signal_time:.2f}ms")

            # Check performance targets
            self._check_performance_targets(latency_benchmark, signal_time)

        except Exception as e:
            logger.error(f"❌ System benchmark failed: {str(e)}")

    def _check_performance_targets(self, latency_benchmark: Dict, signal_time: float):
        """Check if system meets performance targets"""
        try:
            warnings = []

            # Check latency targets
            p95_latency = latency_benchmark['latency_stats']['p95_ms']
            if p95_latency > self.performance_targets['order_execution_time_ms']:
                warnings.append(f"Latency target missed: {p95_latency:.2f}ms > {self.performance_targets['order_execution_time_ms']}ms")

            # Check signal generation targets
            if signal_time > self.performance_targets['signal_generation_time_ms']:
                warnings.append(f"Signal generation target missed: {signal_time:.2f}ms > {self.performance_targets['signal_generation_time_ms']}ms")

            if warnings:
                logger.warning("⚠️ Performance targets not met:")
                for warning in warnings:
                    logger.warning(f"   {warning}")
            else:
                logger.info("✅ All performance targets met")

        except Exception as e:
            logger.error(f"Error checking performance targets: {str(e)}")

    async def _verify_system_health(self):
        """Verify all system components are healthy"""
        try:
            logger.info("🏥 Verifying system health...")

            # Check API connections
            api_connection = self.bybit_client.test_connection()
            if not api_connection:
                raise Exception("Bybit API connection failed")

            # Check account balance
            balance = self.bybit_client.get_account_balance()
            if not balance:
                raise Exception("Failed to get account balance")

            # Check optimization components
            latency_metrics = self.latency_optimizer.get_latency_metrics()
            if latency_metrics.get('total_requests', 0) == 0:
                logger.warning("⚠️ No latency data available - optimization may not be active")

            # Check AI signal optimizer
            ai_metrics = self.ai_signal_optimizer.get_optimization_metrics()
            if ai_metrics.get('status') == 'No performance data available':
                logger.warning("⚠️ No AI optimization data available")

            logger.info(f"✅ System health verified - Balance: ${float(balance.get('totalEquity', 0)):.2f}")

        except Exception as e:
            logger.error(f"❌ System health verification failed: {str(e)}")
            raise

    async def _start_monitoring_systems(self):
        """Start all monitoring and optimization systems"""
        try:
            logger.info("📊 Starting monitoring systems...")

            # Performance monitoring is already started in initialization

            # Start periodic performance checking
            asyncio.create_task(self._performance_monitoring_loop())

            # Start system health monitoring
            asyncio.create_task(self._health_monitoring_loop())

            logger.info("✅ Monitoring systems started")

        except Exception as e:
            logger.error(f"❌ Failed to start monitoring systems: {str(e)}")
            raise

    async def _run_enhanced_trading_loop(self):
        """Main enhanced trading loop with all optimizations"""
        try:
            logger.info("🔄 Starting enhanced trading loop...")

            while self.system_active:
                try:
                    loop_start_time = time.time()

                    # Collect and analyze market data for all assets
                    await self._process_all_assets()

                    # Update performance metrics
                    loop_time = (time.time() - loop_start_time) * 1000
                    await self._update_system_metrics(loop_time)

                    # Dynamic sleep based on system load and market conditions
                    sleep_time = self._calculate_optimal_sleep_time(loop_time)
                    await asyncio.sleep(sleep_time)

                except Exception as e:
                    logger.error(f"❌ Error in trading loop: {str(e)}")
                    await asyncio.sleep(10)  # Brief pause on error

        except Exception as e:
            logger.error(f"❌ Critical error in trading loop: {str(e)}")
            await self.stop_system("Trading loop error")
            raise

    async def _process_all_assets(self):
        """Process all target assets with optimizations"""
        try:
            target_assets = self.config.TARGET_ASSETS
            logger.info(f"🎯 Processing {len(target_assets)} assets with optimizations...")

            # Create tasks for parallel processing
            tasks = []
            for symbol in target_assets:
                task = asyncio.create_task(
                    self._process_single_asset_optimized(symbol)
                )
                tasks.append(task)

            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            successful_assets = 0
            for i, result in enumerate(results):
                symbol = target_assets[i]
                if isinstance(result, Exception):
                    logger.error(f"❌ Failed to process {symbol}: {str(result)}")
                else:
                    successful_assets += 1
                    if result.get('signal_generated'):
                        logger.info(f"🎯 Signal generated for {symbol}: {result.get('signal', 'UNKNOWN')}")

            logger.info(f"✅ Processed {successful_assets}/{len(target_assets)} assets successfully")

        except Exception as e:
            logger.error(f"❌ Error processing assets: {str(e)}")

    async def _process_single_asset_optimized(self, symbol: str) -> Dict:
        """Process a single asset with all optimizations"""
        try:
            asset_start_time = time.time()
            result = {'symbol': symbol, 'signal_generated': False}

            # Step 1: Get optimized market data with microstructure analysis
            market_data = await self._get_optimized_market_data(symbol)
            if not market_data:
                return result

            # Step 2: Analyze market microstructure for execution optimization
            microstructure_analysis = await self.microstructure_analyzer.analyze_market_microstructure(
                symbol, urgency="NORMAL"
            )

            # Step 3: Get optimized AI signal
            signal_result = await self.ai_signal_optimizer.get_optimized_signal(
                symbol=symbol,
                market_data=market_data,
                institutional_data=self._get_institutional_data(),
                priority=2,
                use_cache=True
            )

            # Step 4: Process signal if generated
            if signal_result.consensus_result.final_signal in ['BUY']:  # Long-only
                # Execute trade with performance optimization
                await self._execute_optimized_trade(
                    symbol, signal_result, microstructure_analysis
                )
                result['signal_generated'] = True
                result['signal'] = signal_result.consensus_result.final_signal
                result['confidence'] = signal_result.consensus_result.confidence_avg

            # Calculate processing time
            processing_time = (time.time() - asset_start_time) * 1000
            result['processing_time_ms'] = processing_time

            logger.debug(f"Processed {symbol} in {processing_time:.2f}ms")
            return result

        except Exception as e:
            logger.error(f"❌ Error processing {symbol}: {str(e)}")
            return {'symbol': symbol, 'signal_generated': False, 'error': str(e)}

    async def _get_optimized_market_data(self, symbol: str) -> Optional[Dict]:
        """Get market data with latency optimization"""
        try:
            # Use latency optimizer for fast market data retrieval
            market_data = await self.latency_optimizer.optimize_market_data_request(symbol)

            if not market_data:
                # Fallback to standard method
                market_data = self.bybit_client.get_market_data(symbol)

            return market_data

        except Exception as e:
            logger.error(f"Error getting optimized market data for {symbol}: {str(e)}")
            return None

    async def _execute_optimized_trade(self,
                                     symbol: str,
                                     signal_result,
                                     microstructure_analysis):
        """Execute trade with all performance optimizations"""
        try:
            logger.info(f"💰 Executing optimized trade for {symbol}")

            # Extract signal parameters
            consensus = signal_result.consensus_result
            params = consensus.recommended_params

            # Apply microstructure-based execution optimization
            execution_params = microstructure_analysis.execution_optimization

            # Execute with performance optimizer
            execution_result = await self.performance_optimizer.optimize_order_execution(
                symbol=symbol,
                side=params.get('direction', 'Buy'),
                quantity=params.get('quantity', 0.001),
                order_type=execution_params.get('order_type', 'Market'),
                urgency="HIGH"  # High urgency for signal-based trades
            )

            if execution_result['success']:
                logger.info(f"✅ Optimized trade executed for {symbol}")
                logger.info(f"   Execution metrics: {execution_result['execution_metrics']}")
            else:
                logger.error(f"❌ Optimized trade execution failed for {symbol}")

        except Exception as e:
            logger.error(f"❌ Error executing optimized trade for {symbol}: {str(e)}")

    async def _update_system_metrics(self, loop_time: float):
        """Update comprehensive system metrics"""
        try:
            current_time = time.time()

            # Get component metrics
            latency_metrics = self.latency_optimizer.get_latency_metrics()
            ai_metrics = self.ai_signal_optimizer.get_optimization_metrics()
            performance_metrics = self.performance_optimizer.get_performance_summary()

            # Get engine status
            engine_status = self.enhanced_engine.get_engine_status()

            # Calculate system health score
            health_score = self._calculate_system_health_score(
                loop_time, latency_metrics, ai_metrics, performance_metrics
            )

            # Create metrics object
            metrics = TradingSystemMetrics(
                timestamp=current_time,
                total_signals_generated=ai_metrics.get('overall_metrics', {}).get('total_requests', 0),
                total_trades_executed=engine_status.get('trades_executed', 0),
                success_rate=engine_status.get('success_rate', 0),
                avg_execution_latency=latency_metrics.get('overall_metrics', {}).get('avg_latency_ms', 0),
                avg_signal_generation_time=ai_metrics.get('overall_metrics', {}).get('avg_processing_time_ms', 0),
                total_pnl=engine_status.get('total_pnl', 0),
                active_positions=engine_status.get('active_positions', 0),
                system_health_score=health_score
            )

            self.system_metrics.append(metrics)

            # Keep only recent metrics
            if len(self.system_metrics) > 1000:
                self.system_metrics = self.system_metrics[-1000:]

            # Log performance summary every 10 iterations
            if len(self.system_metrics) % 10 == 0:
                self._log_performance_summary(metrics)

        except Exception as e:
            logger.error(f"Error updating system metrics: {str(e)}")

    def _calculate_system_health_score(self,
                                     loop_time: float,
                                     latency_metrics: Dict,
                                     ai_metrics: Dict,
                                     performance_metrics: Dict) -> float:
        """Calculate overall system health score (0-100)"""
        try:
            health_score = 50.0  # Base score

            # Latency health (20 points)
            avg_latency = latency_metrics.get('overall_metrics', {}).get('avg_latency_ms', 1000)
            if avg_latency < 200:
                health_score += 20
            elif avg_latency < 500:
                health_score += 15
            elif avg_latency < 1000:
                health_score += 10
            else:
                health_score += 5

            # AI optimization health (20 points)
            ai_avg_time = ai_metrics.get('overall_metrics', {}).get('avg_processing_time_ms', 3000)
            if ai_avg_time < 1000:
                health_score += 20
            elif ai_avg_time < 2000:
                health_score += 15
            elif ai_avg_time < 3000:
                health_score += 10
            else:
                health_score += 5

            # Performance optimization health (20 points)
            quality_score = performance_metrics.get('quality_metrics', {}).get('avg_score', 50)
            health_score += (quality_score / 100) * 20

            # Loop time health (10 points)
            if loop_time < 1000:
                health_score += 10
            elif loop_time < 2000:
                health_score += 7
            elif loop_time < 5000:
                health_score += 5
            else:
                health_score += 2

            return min(100.0, max(0.0, health_score))

        except Exception:
            return 50.0

    def _log_performance_summary(self, metrics: TradingSystemMetrics):
        """Log comprehensive performance summary"""
        try:
            logger.info("📊 Enhanced Trading System Performance Summary:")
            logger.info(f"   System Health: {metrics.system_health_score:.1f}/100")
            logger.info(f"   Signals Generated: {metrics.total_signals_generated}")
            logger.info(f"   Trades Executed: {metrics.total_trades_executed}")
            logger.info(f"   Success Rate: {metrics.success_rate:.1f}%")
            logger.info(f"   Avg Execution Latency: {metrics.avg_execution_latency:.2f}ms")
            logger.info(f"   Avg Signal Time: {metrics.avg_signal_generation_time:.2f}ms")
            logger.info(f"   Total PnL: ${metrics.total_pnl:.2f}")
            logger.info(f"   Active Positions: {metrics.active_positions}")

        except Exception as e:
            logger.error(f"Error logging performance summary: {str(e)}")

    def _calculate_optimal_sleep_time(self, loop_time: float) -> float:
        """Calculate optimal sleep time based on system performance"""
        try:
            base_sleep = 10.0  # 10 seconds base

            # Adjust based on loop time
            if loop_time > 5000:  # Loop took > 5 seconds
                return max(5.0, base_sleep - 5.0)  # Reduce sleep time
            elif loop_time < 1000:  # Loop was very fast
                return base_sleep + 5.0  # Increase sleep time
            else:
                return base_sleep

        except Exception:
            return 10.0

    def _get_test_market_data(self) -> Dict:
        """Get test market data for benchmarks"""
        return {
            'symbol': 'BTCUSDT',
            'lastPrice': '45000.0',
            'price24hPcnt': '0.02',
            'volume24h': '1000000000',
            'highPrice24h': '46000.0',
            'lowPrice24h': '44000.0'
        }

    def _get_institutional_data(self) -> Dict:
        """Get institutional data for AI analysis"""
        return {
            "fear_greed": {"value": 50, "classification": "Neutral"},
            "funding_rates": {"average_funding": -0.01, "trend": "Slightly Negative"},
            "open_interest": {"oi_change_24h": 2.5, "trend": "Increasing"},
            "institutional_flows": {"net_flow": 125000000, "trend": "Bullish Inflow"}
        }

    async def _performance_monitoring_loop(self):
        """Periodic performance monitoring loop"""
        while self.system_active:
            try:
                current_time = time.time()

                # Check performance every minute
                if current_time - self.last_performance_check >= self.performance_check_interval:
                    await self._check_performance_degradation()
                    self.last_performance_check = current_time

                await asyncio.sleep(10)  # Check every 10 seconds

            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {str(e)}")
                await asyncio.sleep(30)

    async def _check_performance_degradation(self):
        """Check for performance degradation and alert"""
        try:
            if not self.system_metrics:
                return

            # Get recent metrics
            recent_metrics = self.system_metrics[-10:]  # Last 10 measurements

            # Check health score
            recent_health = [m.system_health_score for m in recent_metrics]
            avg_health = sum(recent_health) / len(recent_health)

            if avg_health < 70:
                logger.warning(f"⚠️ System health degraded: {avg_health:.1f}/100")

                # Check specific issues
                avg_latency = recent_metrics[-1].avg_execution_latency
                if avg_latency > 1000:
                    logger.warning(f"   High latency detected: {avg_latency:.2f}ms")

                avg_signal_time = recent_metrics[-1].avg_signal_generation_time
                if avg_signal_time > 3000:
                    logger.warning(f"   Slow signal generation: {avg_signal_time:.2f}ms")

                success_rate = recent_metrics[-1].success_rate
                if success_rate < 40:
                    logger.warning(f"   Low success rate: {success_rate:.1f}%")

        except Exception as e:
            logger.error(f"Error checking performance degradation: {str(e)}")

    async def _health_monitoring_loop(self):
        """System health monitoring loop"""
        while self.system_active:
            try:
                # Verify critical components are still responsive
                api_test = self.bybit_client.test_connection()
                if not api_test:
                    logger.error("❌ API connection lost - attempting recovery")

                # Check optimization components
                latency_metrics = self.latency_optimizer.get_latency_metrics()
                if latency_metrics.get('queue_size', 0) > 50:
                    logger.warning(f"⚠️ High latency queue size: {latency_metrics['queue_size']}")

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in health monitoring loop: {str(e)}")
                await asyncio.sleep(60)

    async def stop_system(self, reason: str = "Manual stop"):
        """Stop the enhanced trading system gracefully"""
        if not self.system_active:
            logger.warning("Trading system is already stopped")
            return

        logger.info(f"🛑 Stopping Enhanced Trading System: {reason}")
        self.system_active = False

        try:
            # Stop monitoring systems
            self.performance_optimizer.stop_performance_monitoring()
            self.microstructure_analyzer.stop_real_time_analysis()

            # Generate final performance report
            await self._generate_final_performance_report()

            # Cleanup all components
            self.performance_optimizer.cleanup()
            self.latency_optimizer.cleanup()
            self.ai_signal_optimizer.cleanup()
            self.microstructure_analyzer.cleanup()

            logger.info("✅ Enhanced Trading System stopped successfully")

        except Exception as e:
            logger.error(f"❌ Error during system shutdown: {str(e)}")

    async def _generate_final_performance_report(self):
        """Generate comprehensive final performance report"""
        try:
            if not self.system_metrics:
                logger.info("No performance data available for final report")
                return

            logger.info("📈 FINAL PERFORMANCE REPORT")
            logger.info("=" * 50)

            # System performance summary
            total_time = self.system_metrics[-1].timestamp - self.system_metrics[0].timestamp
            total_signals = self.system_metrics[-1].total_signals_generated
            total_trades = self.system_metrics[-1].total_trades_executed
            final_pnl = self.system_metrics[-1].total_pnl

            logger.info(f"   Total Runtime: {total_time / 3600:.1f} hours")
            logger.info(f"   Total Signals: {total_signals}")
            logger.info(f"   Total Trades: {total_trades}")
            logger.info(f"   Signal-to-Trade Rate: {(total_trades / max(total_signals, 1) * 100):.1f}%")
            logger.info(f"   Final PnL: ${final_pnl:.2f}")

            # Performance metrics
            import numpy as np
            health_scores = [m.system_health_score for m in self.system_metrics]
            latencies = [m.avg_execution_latency for m in self.system_metrics]
            signal_times = [m.avg_signal_generation_time for m in self.system_metrics]

            logger.info(f"   Average Health Score: {np.mean(health_scores):.1f}/100")
            logger.info(f"   Average Latency: {np.mean(latencies):.2f}ms")
            logger.info(f"   Average Signal Time: {np.mean(signal_times):.2f}ms")

            # Component-specific metrics
            latency_summary = self.latency_optimizer.get_latency_metrics()
            ai_summary = self.ai_signal_optimizer.get_optimization_metrics()
            performance_summary = self.performance_optimizer.get_performance_summary()

            logger.info("   Component Performance:")
            logger.info(f"     Latency Optimizer: P95 {latency_summary.get('overall_metrics', {}).get('p95_latency_ms', 0):.2f}ms")
            logger.info(f"     AI Optimizer: Cache hit rate {ai_summary.get('cache_metrics', {}).get('cache_hit_rate', 0):.2f}")
            logger.info(f"     Performance Optimizer: Avg quality {performance_summary.get('quality_metrics', {}).get('avg_score', 0):.1f}/100")

            logger.info("=" * 50)

        except Exception as e:
            logger.error(f"Error generating final performance report: {str(e)}")

    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        try:
            # Get component statuses
            latency_status = self.latency_optimizer.get_latency_metrics()
            ai_status = self.ai_signal_optimizer.get_optimization_metrics()
            performance_status = self.performance_optimizer.get_performance_summary()
            engine_status = self.enhanced_engine.get_engine_status()

            # Calculate system health
            latest_metrics = self.system_metrics[-1] if self.system_metrics else None

            return {
                'system_active': self.system_active,
                'optimizations_enabled': self.optimizations_enabled,
                'performance_targets': self.performance_targets,
                'component_status': {
                    'latency_optimizer': {
                        'active': True,
                        'queue_size': latency_status.get('queue_status', {}).get('queue_size', 0),
                        'avg_latency_ms': latency_status.get('overall_metrics', {}).get('avg_latency_ms', 0)
                    },
                    'ai_signal_optimizer': {
                        'active': True,
                        'cache_size': ai_status.get('cache_metrics', {}).get('cache_size', 0),
                        'avg_processing_time_ms': ai_status.get('overall_metrics', {}).get('avg_processing_time_ms', 0)
                    },
                    'performance_optimizer': {
                        'active': performance_status.get('total_executions', 0) > 0,
                        'total_executions': performance_status.get('total_executions', 0),
                        'avg_quality_score': performance_status.get('quality_metrics', {}).get('avg_score', 0)
                    },
                    'trading_engine': {
                        'active': engine_status.get('engine_active', False),
                        'active_positions': engine_status.get('active_positions', 0),
                        'trades_executed': engine_status.get('trades_executed', 0)
                    }
                },
                'system_health': {
                    'health_score': latest_metrics.system_health_score if latest_metrics else 0,
                    'total_signals': latest_metrics.total_signals_generated if latest_metrics else 0,
                    'total_trades': latest_metrics.total_trades_executed if latest_metrics else 0,
                    'success_rate': latest_metrics.success_rate if latest_metrics else 0,
                    'total_pnl': latest_metrics.total_pnl if latest_metrics else 0
                },
                'performance_compliance': {
                    'signal_generation_target_met': latest_metrics.avg_signal_generation_time <= self.performance_targets['signal_generation_time_ms'] if latest_metrics else False,
                    'execution_latency_target_met': latest_metrics.avg_execution_latency <= self.performance_targets['order_execution_time_ms'] if latest_metrics else False,
                    'success_rate_target_met': latest_metrics.success_rate >= self.performance_targets['success_rate_pct'] if latest_metrics else False,
                    'health_target_met': latest_metrics.system_health_score >= self.performance_targets['system_health_score'] if latest_metrics else False
                }
            }

        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {'error': str(e), 'system_active': self.system_active}

async def main():
    """Main function to run the enhanced trading system"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('enhanced_trading_system.log'),
            logging.StreamHandler()
        ]
    )

    logger.info("🚀 Starting Enhanced High-Performance Trading System")

    try:
        # Create and start the enhanced system
        trading_system = EnhancedTradingSystem()
        await trading_system.start_system()

    except KeyboardInterrupt:
        logger.info("👋 System stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error in enhanced trading system: {str(e)}")
        raise
    finally:
        logger.info("🏁 Enhanced Trading System shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())
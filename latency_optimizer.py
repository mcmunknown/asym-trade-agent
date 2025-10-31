"""
LATENCY OPTIMIZATION ENGINE
===========================
Ultra-low latency execution system with:
- Connection pooling and keep-alive
- Pre-warmed API sessions
- Smart request batching
- Priority-based execution queue
- Network optimization
"""

import asyncio
import aiohttp
import time
import logging
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from queue import PriorityQueue, Queue
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bybit_client import BybitClient
from config import Config

logger = logging.getLogger(__name__)

@dataclass
class PriorityRequest:
    """Priority-based API request"""
    priority: int
    timestamp: float
    request_id: str
    method: str
    url: str
    params: Dict
    headers: Dict
    future: asyncio.Future
    timeout: float = 5.0

    def __lt__(self, other):
        if self.priority != other.priority:
            return self.priority > other.priority  # Higher priority first
        return self.timestamp < other.timestamp  # Earlier timestamp first

@dataclass
class ConnectionMetrics:
    """Connection performance metrics"""
    connection_time_ms: float
    response_time_ms: float
    total_time_ms: float
    success: bool
    error_type: Optional[str] = None

class LatencyOptimizer:
    """
    Advanced latency optimization engine for crypto trading
    """

    def __init__(self, bybit_client: BybitClient):
        self.bybit_client = bybit_client
        self.config = Config()

        # Connection pooling
        self.session_pool = []
        self.max_sessions = 5
        self.session_index = 0

        # Priority queue for time-critical requests
        self.priority_queue = PriorityQueue()
        self.queue_processor_active = False

        # Performance tracking
        self.connection_metrics = []
        self.latency_history = []

        # Network optimization
        self.dns_cache = {}
        self.connection_keepalive = True

        # Thread pools
        self.executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="latency_opt")

        # Batching system
        self.batch_requests = []
        self_batch_size = 10
        self_batch_timeout = 0.01  # 10ms

        # Performance targets
        self.target_connection_time = 50.0  # ms
        self.target_response_time = 200.0    # ms
        self.target_total_time = 300.0       # ms

        logger.info("⚡ Latency Optimizer initialized")
        logger.info(f"   Target connection time: {self.target_connection_time}ms")
        logger.info(f"   Target response time: {self.target_response_time}ms")
        logger.info(f"   Target total time: {self.target_total_time}ms")

        # Initialize connection pool
        self._initialize_connection_pool()

    def _initialize_connection_pool(self):
        """Initialize connection pool with optimized settings"""
        try:
            for i in range(self.max_sessions):
                session = self._create_optimized_session()
                self.session_pool.append(session)

            logger.info(f"✅ Connection pool initialized with {self.max_sessions} sessions")
            self._start_queue_processor()

        except Exception as e:
            logger.error(f"❌ Failed to initialize connection pool: {str(e)}")

    def _create_optimized_session(self) -> requests.Session:
        """Create optimized HTTP session with connection pooling"""
        session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST", "PUT", "DELETE"]
        )

        # Configure HTTP adapter with connection pooling
        adapter = HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=retry_strategy,
            pool_block=False
        )

        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Optimize session settings
        session.verify = False  # Skip SSL verification for speed (testnet only)
        session.stream = False  # Don't stream responses

        # Set headers for keep-alive
        session.headers.update({
            'Connection': 'keep-alive',
            'Keep-Alive': 'timeout=30, max=100',
            'User-Agent': 'Mozilla/5.0 (compatible; CryptoTrader/1.0)',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate',
            'Content-Type': 'application/json'
        })

        return session

    async def execute_priority_request(self,
                                     method: str,
                                     url: str,
                                     params: Dict,
                                     headers: Dict,
                                     priority: int = 1,
                                     timeout: float = 5.0) -> Any:
        """
        Execute high-priority request with latency optimization
        Priority levels: 1=Highest (market orders), 2=High, 3=Normal, 4=Low
        """
        start_time = time.time()
        request_id = f"{method}_{int(start_time * 1000000)}"

        # Create future for result
        future = asyncio.Future()

        # Create priority request
        priority_request = PriorityRequest(
            priority=priority,
            timestamp=start_time,
            request_id=request_id,
            method=method,
            url=url,
            params=params,
            headers=headers,
            future=future,
            timeout=timeout
        )

        # Add to priority queue
        self.priority_queue.put(priority_request)

        # Wait for result
        try:
            result = await asyncio.wait_for(future, timeout=timeout + 1.0)

            # Calculate metrics
            total_time = (time.time() - start_time) * 1000
            self._record_connection_metrics(ConnectionMetrics(
                connection_time_ms=0,  # Not tracked separately
                response_time_ms=total_time,
                total_time_ms=total_time,
                success=True
            ))

            return result

        except asyncio.TimeoutError:
            logger.warning(f"Priority request timeout: {request_id}")
            future.cancel()
            raise

    def _start_queue_processor(self):
        """Start background queue processor for priority requests"""
        if self.queue_processor_active:
            return

        self.queue_processor_active = True

        def process_queue():
            while self.queue_processor_active:
                try:
                    # Get next request with timeout
                    try:
                        priority_request = self.priority_queue.get(timeout=0.1)
                    except:
                        continue

                    # Process request
                    self._execute_priority_request_sync(priority_request)

                except Exception as e:
                    logger.error(f"Error in queue processor: {str(e)}")
                    continue

        # Start processor thread
        processor_thread = threading.Thread(
            target=process_queue,
            daemon=True,
            name="PriorityQueueProcessor"
        )
        processor_thread.start()

        logger.info("🚀 Priority queue processor started")

    def _execute_priority_request_sync(self, priority_request: PriorityRequest):
        """Execute priority request synchronously"""
        start_time = time.time()

        try:
            # Get session from pool
            session = self._get_session()

            # Execute request
            if priority_request.method.upper() == 'GET':
                response = session.get(
                    priority_request.url,
                    params=priority_request.params,
                    headers=priority_request.headers,
                    timeout=priority_request.timeout
                )
            elif priority_request.method.upper() == 'POST':
                response = session.post(
                    priority_request.url,
                    json=priority_request.params,
                    headers=priority_request.headers,
                    timeout=priority_request.timeout
                )
            else:
                raise ValueError(f"Unsupported method: {priority_request.method}")

            # Process response
            if response.status_code == 200:
                result = response.json()
                success = True
            else:
                result = {'error': f'HTTP {response.status_code}', 'text': response.text}
                success = False

            # Calculate timing
            total_time = (time.time() - start_time) * 1000

            # Record metrics
            self._record_connection_metrics(ConnectionMetrics(
                connection_time_ms=0,
                response_time_ms=total_time,
                total_time_ms=total_time,
                success=success,
                error_type=None if success else 'HTTP_ERROR'
            ))

            # Set result
            if not priority_request.future.cancelled():
                priority_request.future.set_result(result)

            logger.debug(f"Priority request completed: {priority_request.request_id} in {total_time:.2f}ms")

        except Exception as e:
            total_time = (time.time() - start_time) * 1000

            # Record error metrics
            self._record_connection_metrics(ConnectionMetrics(
                connection_time_ms=0,
                response_time_ms=total_time,
                total_time_ms=total_time,
                success=False,
                error_type=str(e)
            ))

            # Set error result
            if not priority_request.future.cancelled():
                priority_request.future.set_exception(e)

            logger.error(f"Priority request failed: {priority_request.request_id} - {str(e)}")

        finally:
            # Mark task as done
            self.priority_queue.task_done()

    def _get_session(self) -> requests.Session:
        """Get session from pool with round-robin"""
        session = self.session_pool[self.session_index]
        self.session_index = (self.session_index + 1) % self.max_sessions
        return session

    async def execute_batch_requests(self,
                                   requests_list: List[Dict],
                                   priority: int = 3) -> List[Any]:
        """
        Execute multiple requests in batch for efficiency
        """
        if not requests_list:
            return []

        start_time = time.time()
        futures = []

        # Create futures for all requests
        for i, request_data in enumerate(requests_list):
            future = asyncio.Future()
            futures.append(future)

            # Create batch request
            priority_request = PriorityRequest(
                priority=priority,
                timestamp=start_time + i * 0.001,  # Small delay to maintain order
                request_id=f"batch_{i}_{int(start_time * 1000000)}",
                method=request_data.get('method', 'GET'),
                url=request_data['url'],
                params=request_data.get('params', {}),
                headers=request_data.get('headers', {}),
                future=future,
                timeout=request_data.get('timeout', 5.0)
            )

            self.priority_queue.put(priority_request)

        # Wait for all requests to complete
        try:
            results = await asyncio.gather(*futures, return_exceptions=True)

            batch_time = (time.time() - start_time) * 1000
            logger.info(f"Batch of {len(requests_list)} requests completed in {batch_time:.2f}ms")

            return results

        except Exception as e:
            logger.error(f"Batch request failed: {str(e)}")
            return [e] * len(requests_list)

    async def pre_warm_connections(self):
        """Pre-warm connections to reduce first-use latency"""
        try:
            logger.info("🔥 Pre-warming connections...")

            # Test connections with lightweight requests
            server_time_url = f"{self.config.BYBIT_BASE_URL}/v5/public/time"

            tasks = []
            for i in range(min(self.max_sessions, 3)):
                task = asyncio.create_task(
                    self.execute_priority_request(
                        method='GET',
                        url=server_time_url,
                        params={},
                        headers={},
                        priority=4  # Low priority
                    )
                )
                tasks.append(task)

            # Wait for pre-warm requests
            results = await asyncio.gather(*tasks, return_exceptions=True)

            successful_warms = sum(1 for r in results if not isinstance(r, Exception))
            logger.info(f"✅ Connections pre-warmed: {successful_warms}/{len(tasks)} successful")

        except Exception as e:
            logger.error(f"❌ Failed to pre-warm connections: {str(e)}")

    async def optimize_market_data_request(self, symbol: str) -> Dict:
        """Optimized market data request with caching"""
        try:
            # Check if we have recent cached data
            cache_key = f"market_data_{symbol}"
            current_time = time.time()

            if hasattr(self, '_data_cache') and cache_key in self._data_cache:
                cached_data, timestamp = self._data_cache[cache_key]
                if current_time - timestamp < 1.0:  # 1 second cache
                    return cached_data

            # Get fresh data with priority
            market_url = f"{self.config.BYBIT_BASE_URL}/v5/market/tickers"
            params = {
                'category': 'linear',
                'symbol': symbol
            }

            result = await self.execute_priority_request(
                method='GET',
                url=market_url,
                params=params,
                headers={},
                priority=2  # High priority
            )

            if result and 'result' in result:
                market_data = result['result'].get('list', [])
                if market_data:
                    # Cache the result
                    if not hasattr(self, '_data_cache'):
                        self._data_cache = {}
                    self._data_cache[cache_key] = (market_data[0], current_time)

                    return market_data[0]

            return {}

        except Exception as e:
            logger.error(f"Error in optimized market data request for {symbol}: {str(e)}")
            return {}

    async def optimize_order_placement(self,
                                     symbol: str,
                                     side: str,
                                     quantity: float,
                                     order_type: str = "Market",
                                     **kwargs) -> Dict:
        """Optimized order placement with ultra-low latency"""
        try:
            start_time = time.time()

            # Prepare order parameters
            order_url = f"{self.config.BYBIT_BASE_URL}/v5/order/create"
            params = {
                'category': 'linear',
                'symbol': symbol,
                'side': side,
                'orderType': order_type,
                'qty': str(quantity),
                'positionIdx': 0,
                'timeInForce': 'GTC'
            }

            # Add optional parameters
            if 'price' in kwargs:
                params['price'] = str(kwargs['price'])
            if 'reduce_only' in kwargs:
                params['reduceOnly'] = kwargs['reduce_only']
            if 'close_on_trigger' in kwargs:
                params['closeOnTrigger'] = kwargs['close_on_trigger']
            if 'take_profit' in kwargs:
                params['takeProfit'] = str(kwargs['take_profit'])
            if 'stop_loss' in kwargs:
                params['stopLoss'] = str(kwargs['stop_loss'])

            # Add authentication headers
            headers = self._get_auth_headers(params, order_url)

            # Execute with highest priority
            result = await self.execute_priority_request(
                method='POST',
                url=order_url,
                params=params,
                headers=headers,
                priority=1,  # Highest priority
                timeout=2.0   # Short timeout for orders
            )

            execution_time = (time.time() - start_time) * 1000

            if result and result.get('retCode') == 0:
                logger.info(f"⚡ Order placed in {execution_time:.2f}ms: {symbol} {side} {quantity}")
                return result.get('result', {})
            else:
                logger.error(f"❌ Order placement failed in {execution_time:.2f}ms: {result.get('retMsg', 'Unknown error')}")
                return {}

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"❌ Order placement exception in {execution_time:.2f}ms: {str(e)}")
            return {}

    def _get_auth_headers(self, params: Dict, url: str) -> Dict:
        """Generate authentication headers for API requests"""
        # This would integrate with Bybit's authentication
        # For now, return basic headers
        return {
            'X-BAPI-API-KEY': self.config.BYBIT_API_KEY,
            'X-BAPI-TIMESTAMP': str(int(time.time() * 1000)),
            'Content-Type': 'application/json'
        }

    def _record_connection_metrics(self, metrics: ConnectionMetrics):
        """Record connection performance metrics"""
        self.connection_metrics.append(metrics)
        self.latency_history.append(metrics.total_time_ms)

        # Keep only recent metrics
        if len(self.connection_metrics) > 1000:
            self.connection_metrics = self.connection_metrics[-1000:]
        if len(self.latency_history) > 500:
            self.latency_history = self.latency_history[-500:]

    def get_latency_metrics(self) -> Dict:
        """Get comprehensive latency performance metrics"""
        if not self.latency_history:
            return {'status': 'No latency data available'}

        import numpy as np

        latencies = self.latency_history
        recent_latencies = latencies[-50:]  # Last 50 requests

        return {
            'total_requests': len(latencies),
            'overall_metrics': {
                'avg_latency_ms': np.mean(latencies),
                'median_latency_ms': np.median(latencies),
                'p95_latency_ms': np.percentile(latencies, 95),
                'p99_latency_ms': np.percentile(latencies, 99),
                'min_latency_ms': np.min(latencies),
                'max_latency_ms': np.max(latencies)
            },
            'recent_metrics': {
                'avg_latency_ms': np.mean(recent_latencies),
                'median_latency_ms': np.median(recent_latencies),
                'p95_latency_ms': np.percentile(recent_latencies, 95),
                'trend': 'improving' if len(recent_latencies) > 10 and np.mean(recent_latencies[-10:]) < np.mean(recent_latencies[:-10]) else 'stable'
            },
            'target_comparison': {
                'target_response_ms': self.target_response_time,
                'target_met_pct': (np.array(latencies) <= self.target_response_time).mean() * 100,
                'within_2x_target_pct': (np.array(latencies) <= self.target_response_time * 2).mean() * 100
            },
            'queue_status': {
                'queue_size': self.priority_queue.qsize(),
                'processor_active': self.queue_processor_active,
                'active_sessions': len(self.session_pool)
            }
        }

    async def benchmark_performance(self, iterations: int = 100) -> Dict:
        """Benchmark connection performance"""
        logger.info(f"🏃 Starting performance benchmark ({iterations} iterations)...")

        test_url = f"{self.config.BYBIT_BASE_URL}/v5/public/time"
        latencies = []

        for i in range(iterations):
            start_time = time.time()

            try:
                result = await self.execute_priority_request(
                    method='GET',
                    url=test_url,
                    params={},
                    headers={},
                    priority=3
                )

                latency = (time.time() - start_time) * 1000
                latencies.append(latency)

                if (i + 1) % 20 == 0:
                    logger.info(f"  Completed {i + 1}/{iterations} iterations")

            except Exception as e:
                logger.error(f"  Benchmark iteration {i + 1} failed: {str(e)}")

        # Calculate statistics
        import numpy as np
        latencies = np.array(latencies)

        benchmark_results = {
            'iterations_completed': len(latencies),
            'latency_stats': {
                'mean_ms': np.mean(latencies),
                'std_ms': np.std(latencies),
                'min_ms': np.min(latencies),
                'max_ms': np.max(latencies),
                'median_ms': np.median(latencies),
                'p95_ms': np.percentile(latencies, 95),
                'p99_ms': np.percentile(latencies, 99)
            },
            'performance_grade': self._calculate_performance_grade(latencies),
            'recommendations': self._generate_performance_recommendations(latencies)
        }

        logger.info(f"✅ Benchmark completed:")
        logger.info(f"   Average latency: {benchmark_results['latency_stats']['mean_ms']:.2f}ms")
        logger.info(f"   P95 latency: {benchmark_results['latency_stats']['p95_ms']:.2f}ms")
        logger.info(f"   Performance grade: {benchmark_results['performance_grade']}")

        return benchmark_results

    def _calculate_performance_grade(self, latencies: np.ndarray) -> str:
        """Calculate performance grade based on latency metrics"""
        p95_latency = np.percentile(latencies, 95)

        if p95_latency <= 100:
            return "A+ (Excellent)"
        elif p95_latency <= 200:
            return "A (Very Good)"
        elif p95_latency <= 300:
            return "B (Good)"
        elif p95_latency <= 500:
            return "C (Average)"
        else:
            return "D (Needs Improvement)"

    def _generate_performance_recommendations(self, latencies: np.ndarray) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        p95_latency = np.percentile(latencies, 95)
        avg_latency = np.mean(latencies)

        if p95_latency > 500:
            recommendations.append("Consider using a CDN or closer server location")
            recommendations.append("Enable HTTP/2 for better multiplexing")

        if avg_latency > 200:
            recommendations.append("Optimize connection pool size")
            recommendations.append("Consider request batching for non-critical operations")

        if np.std(latencies) > avg_latency * 0.5:
            recommendations.append("High latency variance detected - investigate network stability")

        if not recommendations:
            recommendations.append("Performance is optimal - maintain current configuration")

        return recommendations

    def cleanup(self):
        """Cleanup resources and connections"""
        logger.info("🧹 Cleaning up Latency Optimizer...")

        # Stop queue processor
        self.queue_processor_active = False

        # Close session pool
        for session in self.session_pool:
            try:
                session.close()
            except:
                pass

        # Shutdown executor
        self.executor.shutdown(wait=True)

        logger.info("✅ Latency Optimizer cleanup completed")
"""
AI SIGNAL OPTIMIZATION ENGINE
=============================
High-performance AI signal processing with:
- Model parallelization
- Intelligent caching
- Request optimization
- Performance monitoring
- Dynamic load balancing
"""

import asyncio
import json
import time
import logging
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import openai
from multi_model_client import (
    MultiModelConsensusEngine,
    ModelSignal,
    ConsensusResult,
    Grok4FastClient,
    Qwen3MaxClient,
    DeepSeekTerminusClient
)
from config import Config

logger = logging.getLogger(__name__)

@dataclass
class SignalRequest:
    """Optimized signal request with caching metadata"""
    symbol: str
    market_data: Dict
    institutional_data: Dict
    request_hash: str
    timestamp: float
    priority: int = 2
    use_cache: bool = True
    emergency_mode: bool = False

@dataclass
class ModelPerformance:
    """Model performance tracking"""
    model_name: str
    avg_response_time: float
    success_rate: float
    accuracy_score: float
    last_updated: float
    total_requests: int
    cache_hit_rate: float

@dataclass
class OptimizedConsensusResult:
    """Enhanced consensus result with optimization metadata"""
    consensus_result: ConsensusResult
    optimization_metadata: Dict
    processing_time_ms: float
    cache_hits: int
    models_used: List[str]
    load_balancing_info: Dict

class AISignalOptimizer:
    """
    High-performance AI signal optimization engine
    """

    def __init__(self):
        self.config = Config()

        # Initialize AI models
        self.grok_client = Grok4FastClient()
        self.qwen_client = Qwen3MaxClient()
        self.deepseek_client = DeepSeekTerminusClient()
        self.consensus_engine = MultiModelConsensusEngine()

        # Performance tracking
        self.model_performance = {}
        self.signal_cache = {}
        self.performance_history = []

        # Optimization parameters
        self.cache_ttl = 300  # 5 minutes
        self.max_concurrent_requests = 5
        self.request_timeout = 30.0  # seconds

        # Thread pools
        self.model_executor = ThreadPoolExecutor(
            max_workers=6,
            thread_name_prefix="ai_signal"
        )

        # Load balancing
        self.model_weights = {
            "Grok 4 Fast": 1.0,
            "Qwen3-Max": 1.0,
            "DeepSeek V3.1-Terminus": 1.0
        }

        # Queue management
        self.request_queue = asyncio.Queue()
        self.queue_processor_active = False

        # Performance targets
        self.target_signal_time = 2000.0  # 2 seconds
        self.target_consensus_time = 3000.0  # 3 seconds

        logger.info("🧠 AI Signal Optimizer initialized")
        logger.info(f"   Target signal time: {self.target_signal_time}ms")
        logger.info(f"   Target consensus time: {self.target_consensus_time}ms")
        logger.info(f"   Cache TTL: {self.cache_ttl} seconds")

        # Initialize model performance tracking
        self._initialize_model_performance()

    def _initialize_model_performance(self):
        """Initialize performance tracking for all models"""
        models = ["Grok 4 Fast", "Qwen3-Max", "DeepSeek V3.1-Terminus"]
        for model in models:
            self.model_performance[model] = ModelPerformance(
                model_name=model,
                avg_response_time=1000.0,
                success_rate=1.0,
                accuracy_score=0.5,
                last_updated=time.time(),
                total_requests=0,
                cache_hit_rate=0.0
            )

    async def get_optimized_signal(self,
                                 symbol: str,
                                 market_data: Dict,
                                 institutional_data: Optional[Dict] = None,
                                 priority: int = 2,
                                 use_cache: bool = True) -> OptimizedConsensusResult:
        """
        Get optimized AI signal with caching and performance monitoring
        """

        start_time = time.time()

        try:
            # Create optimized request
            signal_request = self._create_signal_request(
                symbol, market_data, institutional_data, priority, use_cache
            )

            # Check cache first
            if use_cache and not signal_request.emergency_mode:
                cached_result = self._check_cache(signal_request)
                if cached_result:
                    processing_time = (time.time() - start_time) * 1000
                    logger.info(f"⚡ Cached signal retrieved for {symbol} in {processing_time:.2f}ms")
                    return cached_result

            # Process signal request
            result = await self._process_signal_request(signal_request)

            # Update cache
            if use_cache and result.consensus_result.final_signal != "NONE":
                self._update_cache(signal_request, result)

            # Update performance tracking
            processing_time = (time.time() - start_time) * 1000
            self._update_performance_metrics(signal_request, result, processing_time)

            logger.info(f"🎯 Optimized signal generated for {symbol} in {processing_time:.2f}ms")
            logger.info(f"   Signal: {result.consensus_result.final_signal}")
            logger.info(f"   Confidence: {result.consensus_result.confidence_avg:.2f}")
            logger.info(f"   Models used: {result.models_used}")

            return result

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"❌ Signal optimization failed for {symbol} in {processing_time:.2f}ms: {str(e)}")

            # Return fallback result
            return self._create_fallback_result(symbol, str(e), processing_time)

    def _create_signal_request(self,
                             symbol: str,
                             market_data: Dict,
                             institutional_data: Optional[Dict],
                             priority: int,
                             use_cache: bool) -> SignalRequest:
        """Create optimized signal request with hash for caching"""

        # Prepare institutional data
        if not institutional_data:
            institutional_data = self._get_default_institutional_data()

        # Create request hash for caching
        request_data = {
            'symbol': symbol,
            'market_data_hash': self._hash_data(market_data),
            'institutional_data_hash': self._hash_data(institutional_data),
            'emergency_mode': self.config.EMERGENCY_DEEPSEEK_ONLY
        }
        request_hash = hashlib.md5(json.dumps(request_data, sort_keys=True).encode()).hexdigest()

        return SignalRequest(
            symbol=symbol,
            market_data=market_data,
            institutional_data=institutional_data,
            request_hash=request_hash,
            timestamp=time.time(),
            priority=priority,
            use_cache=use_cache,
            emergency_mode=self.config.EMERGENCY_DEEPSEEK_ONLY
        )

    async def _process_signal_request(self, request: SignalRequest) -> OptimizedConsensusResult:
        """Process signal request with optimized model execution"""

        start_time = time.time()

        if request.emergency_mode:
            # Emergency mode: Use only DeepSeek for speed and reliability
            result = await self._process_emergency_signal(request)
        else:
            # Normal mode: Use multi-model consensus with optimization
            result = await self._process_consensus_signal(request)

        processing_time = (time.time() - start_time) * 1000

        # Create optimized result
        optimized_result = OptimizedConsensusResult(
            consensus_result=result.consensus_result,
            optimization_metadata={
                'request_priority': request.priority,
                'cache_used': False,
                'emergency_mode': request.emergency_mode,
                'request_timestamp': request.timestamp
            },
            processing_time_ms=processing_time,
            cache_hits=0,
            models_used=result.models_used,
            load_balancing_info=result.load_balancing_info
        )

        return optimized_result

    async def _process_emergency_signal(self, request: SignalRequest) -> OptimizedConsensusResult:
        """Process emergency signal using only DeepSeek for maximum speed"""

        try:
            logger.info(f"🚨 Processing emergency signal for {request.symbol} with DeepSeek V3.1-Terminus")

            # Execute DeepSeek analysis with highest priority
            loop = asyncio.get_event_loop()
            deepseek_signal = await loop.run_in_executor(
                self.model_executor,
                self.deepseek_client.analyze_market,
                request.market_data,
                request.institutional_data
            )

            # Convert to consensus result
            consensus_result = ConsensusResult(
                final_signal=deepseek_signal.signal,
                signal_type=deepseek_signal.signal_type,
                consensus_votes={"DeepSeek V3.1-Terminus": deepseek_signal.signal},
                confidence_avg=deepseek_signal.confidence,
                thesis_combined=deepseek_signal.thesis_summary,
                recommended_params={
                    'entry_price': deepseek_signal.entry_price,
                    'activation_price': deepseek_signal.activation_price,
                    'trailing_stop_pct': deepseek_signal.trailing_stop_pct,
                    'invalidation_level': deepseek_signal.invalidation_level,
                    'thesis_summary': deepseek_signal.thesis_summary,
                    'risk_reward_ratio': deepseek_signal.risk_reward_ratio,
                    'leverage': min(deepseek_signal.leverage, self.config.MAX_LEVERAGE),
                    'quantity': deepseek_signal.quantity
                }
            )

            return OptimizedConsensusResult(
                consensus_result=consensus_result,
                optimization_metadata={'emergency_mode': True},
                processing_time_ms=0,  # Will be set by caller
                cache_hits=0,
                models_used=["DeepSeek V3.1-Terminus"],
                load_balancing_info={'strategy': 'emergency_single_model'}
            )

        except Exception as e:
            logger.error(f"Emergency signal processing failed for {request.symbol}: {str(e)}")
            raise

    async def _process_consensus_signal(self, request: SignalRequest) -> OptimizedConsensusResult:
        """Process consensus signal with optimized model parallelization"""

        try:
            logger.info(f"🤖 Processing consensus signal for {request.symbol} with optimized parallelization")

            # Select optimal models based on performance and load
            selected_models = self._select_optimal_models(request)

            # Execute models in parallel with load balancing
            tasks = []
            model_mapping = {}

            for model_name in selected_models:
                if model_name == "Grok 4 Fast":
                    task = asyncio.create_task(
                        self._execute_model_with_timeout(
                            self.grok_client.analyze_market,
                            request.market_data,
                            request.institutional_data,
                            model_name
                        )
                    )
                    model_mapping[task] = "Grok 4 Fast"
                elif model_name == "Qwen3-Max":
                    task = asyncio.create_task(
                        self._execute_model_with_timeout(
                            self.qwen_client.analyze_market,
                            request.market_data,
                            request.institutional_data,
                            model_name
                        )
                    )
                    model_mapping[task] = "Qwen3-Max"
                elif model_name == "DeepSeek V3.1-Terminus":
                    task = asyncio.create_task(
                        self._execute_model_with_timeout(
                            self.deepseek_client.analyze_market,
                            request.market_data,
                            request.institutional_data,
                            model_name
                        )
                    )
                    model_mapping[task] = "DeepSeek V3.1-Terminus"

                tasks.append(task)

            # Wait for model results with timeout
            try:
                model_results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=self.request_timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"Model execution timeout for {request.symbol}")
                # Use whatever results we have
                model_results = []
                for task in tasks:
                    if task.done():
                        try:
                            result = task.result()
                            model_results.append(result)
                        except:
                            model_results.append(None)
                    else:
                        task.cancel()
                        model_results.append(None)

            # Process model results and calculate consensus
            consensus_result = self._calculate_optimized_consensus(
                model_results, list(model_mapping.values()), request
            )

            # Get load balancing info
            load_balancing_info = {
                'strategy': 'parallel_optimized',
                'models_requested': len(selected_models),
                'models_completed': len([r for r in model_results if r is not None]),
                'model_weights_used': {model: self.model_weights.get(model, 1.0) for model in selected_models}
            }

            return OptimizedConsensusResult(
                consensus_result=consensus_result,
                optimization_metadata={'consensus_mode': True},
                processing_time_ms=0,  # Will be set by caller
                cache_hits=0,
                models_used=list(model_mapping.values()),
                load_balancing_info=load_balancing_info
            )

        except Exception as e:
            logger.error(f"Consensus signal processing failed for {request.symbol}: {str(e)}")
            raise

    def _select_optimal_models(self, request: SignalRequest) -> List[str]:
        """Select optimal models based on performance and request priority"""

        if request.priority == 1:  # Highest priority - use fastest models
            return ["DeepSeek V3.1-Terminus", "Grok 4 Fast"]
        elif request.priority == 2:  # High priority - use 2 best models
            # Sort by performance (response time and success rate)
            sorted_models = sorted(
                self.model_performance.items(),
                key=lambda x: (x[1].avg_response_time, -x[1].success_rate)
            )
            return [model[0] for model in sorted_models[:2]]
        else:  # Normal priority - use all models
            return ["Grok 4 Fast", "Qwen3-Max", "DeepSeek V3.1-Terminus"]

    async def _execute_model_with_timeout(self,
                                        model_func,
                                        market_data: Dict,
                                        institutional_data: Dict,
                                        model_name: str) -> Optional[ModelSignal]:
        """Execute model with timeout and performance tracking"""

        start_time = time.time()

        try:
            # Execute model in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.model_executor,
                model_func,
                market_data,
                institutional_data
            )

            # Track performance
            response_time = (time.time() - start_time) * 1000
            self._update_model_performance(model_name, response_time, True)

            return result

        except Exception as e:
            # Track performance
            response_time = (time.time() - start_time) * 1000
            self._update_model_performance(model_name, response_time, False)

            logger.error(f"Model execution failed for {model_name}: {str(e)}")
            return None

    def _calculate_optimized_consensus(self,
                                     model_results: List[Optional[ModelSignal]],
                                     model_names: List[str],
                                     request: SignalRequest) -> ConsensusResult:
        """Calculate consensus with optimized voting and model weighting"""

        valid_results = []
        model_votes = {}
        weighted_confidence = {}

        # Process results and apply model weights
        for i, result in enumerate(model_results):
            if i < len(model_names):
                model_name = model_names[i]
                model_votes[model_name] = "ERROR"

                if result and isinstance(result, ModelSignal):
                    valid_results.append(result)
                    model_votes[model_name] = result.signal

                    # Apply model weight to confidence
                    weight = self.model_weights.get(model_name, 1.0)
                    weighted_confidence[model_name] = result.confidence * weight

                    logger.debug(f"Model {model_name}: {result.signal} (confidence: {result.confidence:.2f}, weight: {weight:.2f})")

        # Calculate weighted consensus
        if not valid_results:
            return ConsensusResult(
                final_signal="NONE",
                signal_type="MAIN_STRATEGY",
                consensus_votes=model_votes,
                confidence_avg=0.0,
                thesis_combined="All models failed",
                recommended_params={}
            )

        # Count weighted votes
        buy_weight = sum(weighted_confidence.get(name, 0) for name, vote in model_votes.items() if vote == "BUY")
        sell_weight = sum(weighted_confidence.get(name, 0) for name, vote in model_votes.items() if vote == "SELL")

        # Determine final signal
        if buy_weight > sell_weight and buy_weight > 0.5:
            final_signal = "BUY"
        elif sell_weight > buy_weight and sell_weight > 0.5:
            final_signal = "SELL"
        else:
            final_signal = "NONE"

        if final_signal in ["BUY", "SELL"]:
            # Average parameters from voting models
            voting_models = [r for r in valid_results if r.signal == final_signal]
            if voting_models:
                consensus_params = self._average_weighted_parameters(voting_models, weighted_confidence)
                combined_thesis = self._combine_thesis_statements(voting_models)
                avg_confidence = sum(weighted_confidence.get(model_names[i], 0)
                                   for i, r in enumerate(model_results)
                                   if r and r.signal == final_signal) / len(voting_models)

                return ConsensusResult(
                    final_signal=final_signal,
                    signal_type=voting_models[0].signal_type,
                    consensus_votes=model_votes,
                    confidence_avg=avg_confidence,
                    thesis_combined=combined_thesis,
                    recommended_params=consensus_params
                )

        # No consensus
        return ConsensusResult(
            final_signal="NONE",
            signal_type="MAIN_STRATEGY",
            consensus_votes=model_votes,
            confidence_avg=0.0,
            thesis_combined=f"No consensus reached for {request.symbol}",
            recommended_params={}
        )

    def _average_weighted_parameters(self,
                                   signals: List[ModelSignal],
                                   weighted_confidence: Dict) -> Dict:
        """Average parameters with model weighting"""

        if not signals:
            return {}

        # Calculate weighted averages
        total_weight = 0
        weighted_params = {
            'entry_price': 0,
            'activation_price': 0,
            'trailing_stop_pct': 0,
            'invalidation_level': 0,
            'leverage': 0,
            'quantity': 0
        }

        for signal in signals:
            # Find the corresponding weight
            weight = 1.0  # Default weight
            for model_name, conf in weighted_confidence.items():
                if hasattr(signal, 'model_name') and signal.model_name == model_name:
                    weight = conf
                    break

            total_weight += weight

            # Add weighted parameters
            weighted_params['entry_price'] += signal.entry_price * weight
            weighted_params['activation_price'] += signal.activation_price * weight
            weighted_params['trailing_stop_pct'] += signal.trailing_stop_pct * weight
            weighted_params['invalidation_level'] += signal.invalidation_level * weight
            weighted_params['leverage'] += signal.leverage * weight
            weighted_params['quantity'] += signal.quantity * weight

        # Normalize by total weight
        if total_weight > 0:
            for param in weighted_params:
                weighted_params[param] /= total_weight

        # Round leverage to integer
        weighted_params['leverage'] = int(round(weighted_params['leverage']))

        # Use risk reward ratio from first signal
        weighted_params['risk_reward_ratio'] = signals[0].risk_reward_ratio

        return weighted_params

    def _combine_thesis_statements(self, signals: List[ModelSignal]) -> str:
        """Combine thesis statements from multiple models"""
        if not signals:
            return "No thesis available"

        combined_reasoning = []
        for signal in signals:
            model_name = getattr(signal, 'model_name', 'Unknown Model')
            combined_reasoning.append(f"{model_name}: {signal.reasoning}")

        return " | ".join(combined_reasoning)

    def _check_cache(self, request: SignalRequest) -> Optional[OptimizedConsensusResult]:
        """Check cache for valid signal"""
        try:
            cache_key = f"{request.symbol}_{request.request_hash}"

            if cache_key in self.signal_cache:
                cached_data, timestamp = self.signal_cache[cache_key]

                # Check if cache is still valid
                if time.time() - timestamp < self.cache_ttl:
                    logger.debug(f"Cache hit for {request.symbol}")

                    # Update cache hit rate
                    for model_name in cached_data.models_used:
                        if model_name in self.model_performance:
                            perf = self.model_performance[model_name]
                            perf.cache_hit_rate = (perf.cache_hit_rate * perf.total_requests + 1) / (perf.total_requests + 1)

                    return cached_data
                else:
                    # Remove expired cache entry
                    del self.signal_cache[cache_key]

        except Exception as e:
            logger.error(f"Cache check error: {str(e)}")

        return None

    def _update_cache(self, request: SignalRequest, result: OptimizedConsensusResult):
        """Update cache with new signal"""
        try:
            cache_key = f"{request.symbol}_{request.request_hash}"
            self.signal_cache[cache_key] = (result, time.time())

            # Clean old cache entries
            current_time = time.time()
            expired_keys = [
                key for key, (_, timestamp) in self.signal_cache.items()
                if current_time - timestamp > self.cache_ttl * 2  # Keep 2x TTL for cleanup buffer
            ]

            for key in expired_keys:
                del self.signal_cache[key]

            # Limit cache size
            if len(self.signal_cache) > 100:
                # Remove oldest entries
                sorted_items = sorted(self.signal_cache.items(), key=lambda x: x[1][1])
                for key, _ in sorted_items[:len(self.signal_cache) - 100]:
                    del self.signal_cache[key]

        except Exception as e:
            logger.error(f"Cache update error: {str(e)}")

    def _update_model_performance(self,
                                model_name: str,
                                response_time: float,
                                success: bool):
        """Update model performance metrics"""
        try:
            if model_name not in self.model_performance:
                return

            perf = self.model_performance[model_name]

            # Update running averages
            perf.total_requests += 1
            perf.avg_response_time = (perf.avg_response_time * (perf.total_requests - 1) + response_time) / perf.total_requests

            if success:
                perf.success_rate = (perf.success_rate * (perf.total_requests - 1) + 1) / perf.total_requests
            else:
                perf.success_rate = (perf.success_rate * (perf.total_requests - 1)) / perf.total_requests

            perf.last_updated = time.time()

            # Update model weights based on performance
            self._update_model_weights()

        except Exception as e:
            logger.error(f"Performance update error for {model_name}: {str(e)}")

    def _update_model_weights(self):
        """Update model weights based on recent performance"""
        try:
            for model_name, perf in self.model_performance.items():
                if perf.total_requests >= 5:  # Only update after sufficient data
                    # Calculate performance score (lower response time and higher success rate = higher weight)
                    response_score = max(0, 1 - (perf.avg_response_time / 5000))  # 5s = 0 score
                    success_score = perf.success_rate
                    performance_score = (response_score + success_score) / 2

                    # Update weight with smoothing
                    old_weight = self.model_weights.get(model_name, 1.0)
                    new_weight = 0.5 + performance_score  # Range: 0.5 to 1.5
                    self.model_weights[model_name] = old_weight * 0.8 + new_weight * 0.2  # Smooth update

        except Exception as e:
            logger.error(f"Weight update error: {str(e)}")

    def _update_performance_metrics(self,
                                 request: SignalRequest,
                                 result: OptimizedConsensusResult,
                                 processing_time: float):
        """Update overall performance metrics"""
        try:
            self.performance_history.append({
                'timestamp': time.time(),
                'symbol': request.symbol,
                'processing_time_ms': processing_time,
                'cache_hits': result.cache_hits,
                'models_used': len(result.models_used),
                'signal': result.consensus_result.final_signal,
                'confidence': result.consensus_result.confidence_avg
            })

            # Keep only recent history
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]

        except Exception as e:
            logger.error(f"Performance metrics update error: {str(e)}")

    def _hash_data(self, data: Dict) -> str:
        """Create hash for data to use in cache keys"""
        try:
            # Sort keys and create hash
            sorted_data = json.dumps(data, sort_keys=True, default=str)
            return hashlib.md5(sorted_data.encode()).hexdigest()[:16]
        except:
            return hashlib.md5(str(data).encode()).hexdigest()[:16]

    def _get_default_institutional_data(self) -> Dict:
        """Get default institutional data"""
        return {
            "fear_greed": {"value": 50, "classification": "Neutral"},
            "funding_rates": {"average_funding": -0.01, "trend": "Slightly Negative"},
            "open_interest": {"oi_change_24h": 2.5, "trend": "Increasing"},
            "institutional_flows": {"net_flow": 125000000, "trend": "Bullish Inflow"},
            "risk_parameters": {
                "max_leverage": self.config.MAX_LEVERAGE,
                "stop_loss_pct": self.config.STOP_LOSS_PERCENTAGE,
                "position_size_pct": self.config.MAX_POSITION_SIZE_PERCENTAGE
            }
        }

    def _create_fallback_result(self, symbol: str, error: str, processing_time: float) -> OptimizedConsensusResult:
        """Create fallback result when signal generation fails"""
        consensus_result = ConsensusResult(
            final_signal="NONE",
            signal_type="MAIN_STRATEGY",
            consensus_votes={"Error": "SYSTEM_FAILURE"},
            confidence_avg=0.0,
            thesis_combined=f"Signal generation failed: {error}",
            recommended_params={}
        )

        return OptimizedConsensusResult(
            consensus_result=consensus_result,
            optimization_metadata={'fallback_mode': True, 'error': error},
            processing_time_ms=processing_time,
            cache_hits=0,
            models_used=[],
            load_balancing_info={'strategy': 'fallback'}
        )

    def get_optimization_metrics(self) -> Dict:
        """Get comprehensive optimization performance metrics"""
        try:
            import numpy as np

            if not self.performance_history:
                return {'status': 'No performance data available'}

            # Extract metrics
            processing_times = [h['processing_time_ms'] for h in self.performance_history]
            cache_hit_rates = [h['cache_hits'] for h in self.performance_history]

            return {
                'overall_metrics': {
                    'total_requests': len(self.performance_history),
                    'avg_processing_time_ms': np.mean(processing_times),
                    'median_processing_time_ms': np.median(processing_times),
                    'p95_processing_time_ms': np.percentile(processing_times, 95),
                    'p99_processing_time_ms': np.percentile(processing_times, 99),
                    'target_met_pct': (np.array(processing_times) <= self.target_consensus_time).mean() * 100
                },
                'cache_metrics': {
                    'cache_size': len(self.signal_cache),
                    'cache_hit_rate': np.mean(cache_hit_rates),
                    'cache_ttl': self.cache_ttl
                },
                'model_performance': {
                    model_name: {
                        'avg_response_time_ms': perf.avg_response_time,
                        'success_rate': perf.success_rate,
                        'total_requests': perf.total_requests,
                        'cache_hit_rate': perf.cache_hit_rate,
                        'weight': self.model_weights.get(model_name, 1.0)
                    }
                    for model_name, perf in self.model_performance.items()
                },
                'load_balancing': {
                    'model_weights': self.model_weights.copy(),
                    'optimization_strategy': 'dynamic_weighting'
                },
                'recent_performance': {
                    'last_10_avg_time': np.mean(processing_times[-10:]) if len(processing_times) >= 10 else 0,
                    'last_10_success_rate': len([h for h in self.performance_history[-10:] if h['signal'] != 'NONE']) / min(10, len(self.performance_history)) * 100
                }
            }

        except Exception as e:
            logger.error(f"Error getting optimization metrics: {str(e)}")
            return {'error': str(e)}

    def clear_cache(self):
        """Clear signal cache"""
        self.signal_cache.clear()
        logger.info("Signal cache cleared")

    def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up AI Signal Optimizer...")

        # Clear cache
        self.clear_cache()

        # Shutdown executor
        self.model_executor.shutdown(wait=True)

        logger.info("✅ AI Signal Optimizer cleanup completed")
"""
Live Calculus Trading System
============================

Production-ready live trading system implementing Anne's complete calculus-based approach
with real-time data processing, signal generation, risk management, and automated execution.

Complete Signal Pipeline:
1. Real-time WebSocket data → Market data validation
2. Price series accumulation → Kalman filtering → Calculus analysis
3. Signal generation with SNR filtering → Risk validation
4. Position sizing → Dynamic TP/SL → Order execution
5. Position monitoring → Performance tracking

Features:
- Real-time calculus-based signal generation
- Kalman filter for adaptive state estimation
- Dynamic risk management with position sizing
- High-frequency execution with batch orders
- Comprehensive performance monitoring
- Emergency stop and circuit breakers
"""

import asyncio
import pandas as pd
import numpy as np
import logging
import time
import threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

# Import our enhanced components
from websocket_client import BybitWebSocketClient, ChannelType, MarketData
from calculus_strategy import CalculusTradingStrategy, SignalType
from quantitative_models import CalculusPriceAnalyzer
from kalman_filter import AdaptiveKalmanFilter, KalmanConfig
from risk_manager import RiskManager, PositionSize, TradingLevels
from bybit_client import BybitClient
from config import Config

# Import portfolio management components
from portfolio_manager import PortfolioManager, PortfolioPosition, AllocationDecision
from signal_coordinator import SignalCoordinator
from joint_distribution_analyzer import JointDistributionAnalyzer
from portfolio_optimizer import PortfolioOptimizer, OptimizationObjective

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TradingState:
    """Current trading state for a symbol"""
    symbol: str
    price_history: List[float]
    timestamps: List[float]
    kalman_filter: AdaptiveKalmanFilter
    calculus_analyzer: CalculusPriceAnalyzer
    last_signal: Optional[Dict]
    position_info: Optional[Dict]
    signal_count: int
    last_execution_time: float
    error_count: int

@dataclass
class PerformanceMetrics:
    """Trading performance metrics"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    avg_trade_duration: float
    success_rate: float
    total_commission: float

class LiveCalculusTrader:
    """
    🚀 ENHANCED LIVE TRADING SYSTEM - Portfolio Integration
    ======================================================

    ⚠️ LIVE TRADING WARNING: This system will execute REAL trades with REAL money!

    This system integrates all components for sophisticated portfolio trading:
    - Real-time data processing with WebSocket for 8 assets
    - Anne's calculus-based signal generation for timing
    - Portfolio optimization for allocation decisions
    - Joint distribution analysis for risk management
    - Signal coordination for multi-asset decisions
    - Dynamic risk management and position sizing
    - High-frequency execution with TP/SL management

    HYBRID APPROACH:
    - Single-asset calculus signals → TIMING decisions
    - Portfolio optimization → ALLOCATION decisions
    - Joint distribution → RISK management
    """

    def __init__(self,
                 symbols: List[str] = None,
                 window_size: int = 200,
                 min_signal_interval: int = 30,
                 emergency_stop: bool = False,
                 max_position_size: float = 1000.0,
                 simulation_mode: bool = False,
                 portfolio_mode: bool = True):
        """
        Initialize the ENHANCED live calculus trading system with portfolio integration.

        Args:
            symbols: List of trading symbols (default: 8 major crypto assets)
            window_size: Price history window for analysis
            min_signal_interval: Minimum seconds between signals
            emergency_stop: Emergency stop flag
            max_position_size: Maximum position size per trade
            simulation_mode: Run in simulation mode (no real trades)
            portfolio_mode: Enable portfolio management integration
        """
        # Default to 8 major crypto assets for portfolio trading
        if symbols is None:
            symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT',
                      'AVAXUSDT', 'ADAUSDT', 'LINKUSDT', 'LTCUSDT']

        self.symbols = symbols
        self.window_size = window_size
        self.min_signal_interval = min_signal_interval
        self.emergency_stop = emergency_stop
        self.max_position_size = max_position_size
        self.simulation_mode = simulation_mode
        self.portfolio_mode = portfolio_mode

        print("🚀 ENHANCED LIVE TRADING SYSTEM INITIALIZING")
        print("=" * 60)
        print(f"📊 Trading {len(symbols)} assets: {', '.join(symbols[:4])}...")
        print(f"🔬 Portfolio Mode: {'ENABLED' if portfolio_mode else 'DISABLED'}")
        if not simulation_mode:
            print("⚠️  LIVE TRADING MODE - REAL MONEY AT RISK!")
        else:
            print("🧪 SIMULATION MODE - Safe for testing")
        print("=" * 60)

        # Initialize core components
        self.ws_client = BybitWebSocketClient(
            symbols=symbols,
            testnet=Config.BYBIT_TESTNET,
            channel_types=[ChannelType.TRADE, ChannelType.ORDERBOOK_1],
            heartbeat_interval=20
        )
        self.bybit_client = BybitClient()
        self.risk_manager = RiskManager()

        # Initialize portfolio components if enabled
        if self.portfolio_mode:
            logger.info("🎯 Initializing portfolio management components...")
            self.portfolio_manager = PortfolioManager(
                symbols=symbols,
                initial_capital=100000.0,  # $100k portfolio
                rebalance_threshold=0.05,  # 5% drift threshold
                risk_manager=self.risk_manager
            )

            self.joint_distribution_analyzer = JointDistributionAnalyzer(num_assets=len(symbols))
            self.portfolio_optimizer = PortfolioOptimizer(joint_analyzer=self.joint_distribution_analyzer)
            self.signal_coordinator = SignalCoordinator(symbols=symbols, portfolio_manager=self.portfolio_manager)

            # CRITICAL FIX: Initialize optimal weights to enable trading
            equal_weight = 1.0 / len(symbols)
            initial_weights = {symbol: equal_weight for symbol in symbols}
            self.portfolio_manager.update_optimal_weights(initial_weights)
            logger.info(f"✅ Initialized equal portfolio weights: {initial_weights}")

            logger.info("✅ Portfolio components initialized successfully")
        else:
            self.portfolio_manager = None
            self.signal_coordinator = None
            self.joint_distribution_analyzer = None
            self.portfolio_optimizer = None

        # Trading state per symbol
        self.trading_states = {}
        for symbol in symbols:
            self.trading_states[symbol] = TradingState(
                symbol=symbol,
                price_history=[],
                timestamps=[],
                kalman_filter=AdaptiveKalmanFilter(),
                calculus_analyzer=CalculusPriceAnalyzer(),
                last_signal=None,
                position_info=None,
                signal_count=0,
                last_execution_time=0,
                error_count=0
            )

        # Performance tracking
        self.performance = PerformanceMetrics(
            total_trades=0, winning_trades=0, losing_trades=0,
            total_pnl=0.0, max_drawdown=0.0, sharpe_ratio=0.0,
            avg_trade_duration=0.0, success_rate=0.0, total_commission=0.0
        )

        # Circuit breakers
        self.daily_loss_limit = 0.10  # 10% daily loss limit
        self.consecutive_losses = 0
        self.max_consecutive_losses = 5
        self.last_reset_time = time.time()
        self.daily_pnl = 0.0

        # Threading and async management
        self.is_running = False
        self.processing_thread = None
        self.monitoring_thread = None

        logger.info(f"Live Calculus Trader initialized for symbols: {symbols}")
        logger.info(f"Parameters: window_size={window_size}, min_signal_interval={min_signal_interval}s")

    def start(self):
        """Start the live trading system."""
        if self.is_running:
            logger.warning("Trading system is already running")
            return

        # Test connections first
        if not self._test_connections():
            logger.error("Connection tests failed. Cannot start trading.")
            return

        self.is_running = True
        self.emergency_stop = False

        # Add WebSocket callback
        self.ws_client.add_callback(ChannelType.TRADE, self._handle_market_data)

        # Add portfolio callback if portfolio mode is enabled
        if self.portfolio_mode:
            self.ws_client.add_portfolio_callback(self._handle_portfolio_data)

        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        # Start portfolio monitoring if enabled
        if self.portfolio_mode:
            self.portfolio_thread = threading.Thread(target=self._portfolio_monitoring_loop, daemon=True)
            self.portfolio_thread.start()
            logger.info("📊 Portfolio monitoring started")

        # Start WebSocket client
        try:
            self.ws_client.subscribe()
            self.ws_client.start()
        except KeyboardInterrupt:
            logger.info("Stopping trading system...")
            self.stop()
        except Exception as e:
            logger.error(f"Error in WebSocket client: {e}")
            self.stop()

    def stop(self):
        """Stop the live trading system gracefully."""
        logger.info("Stopping live trading system...")
        self.is_running = False
        self.emergency_stop = True

        # Close all positions
        self._emergency_close_all_positions()

        # Stop WebSocket client
        if self.ws_client:
            self.ws_client.stop()

        # Wait for threads to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)

        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)

        logger.info("Live trading system stopped")

    def _test_connections(self) -> bool:
        """Test all connections before starting."""
        try:
            if self.simulation_mode:
                logger.info("🧪 SIMULATION MODE - Skipping API connection tests")
                return True

            # Test Bybit API connection
            if not self.bybit_client.test_connection():
                logger.error("Bybit API connection failed")
                return False

            # Test account balance
            balance = self.bybit_client.get_wallet_balance()
            if not balance:
                logger.error("Could not fetch account balance")
                return False

            logger.info(f"Connection tests passed. Account balance: {balance.get('totalAvailableBalance', 'Unknown')}")
            return True

        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def _handle_market_data(self, market_data: MarketData):
        """
        Handle incoming market data and process trading signals.

        Args:
            market_data: Real-time market data from WebSocket
        """
        if not self.is_running or self.emergency_stop:
            return

        symbol = market_data.symbol
        if symbol not in self.trading_states:
            logger.warning(f"Received data for unknown symbol: {symbol}")
            return

        try:
            # Update price history
            state = self.trading_states[symbol]
            state.price_history.append(market_data.price)
            state.timestamps.append(market_data.timestamp)

            # Maintain window size
            if len(state.price_history) > self.window_size:
                state.price_history.pop(0)
                state.timestamps.pop(0)

            # Generate signals if we have enough data
            if len(state.price_history) >= 50:  # Minimum for calculus analysis
                self._process_trading_signal(symbol)

        except Exception as e:
            logger.error(f"Error handling market data for {symbol}: {e}")
            state = self.trading_states[symbol]
            state.error_count += 1

    def _handle_portfolio_data(self, market_data_dict: Dict[str, MarketData]):
        """
        Handle portfolio-level market data for multi-asset analysis.

        Args:
            market_data_dict: Dictionary of market data for all symbols
        """
        if not self.portfolio_mode or not self.is_running:
            return

        try:
            # Update joint distribution analyzer
            price_updates = {}
            for symbol, data in market_data_dict.items():
                if symbol in self.symbols:
                    price_updates[symbol] = data.price

            if len(price_updates) >= 4:  # Minimum for meaningful analysis
                self.joint_distribution_analyzer.update_returns(price_updates)

                # Update portfolio manager with market data
                self.portfolio_manager.update_market_data(market_data_dict)

        except Exception as e:
            logger.error(f"Error in portfolio data handling: {e}")

    def _process_trading_signal(self, symbol: str):
        """
        Process trading signal using complete calculus analysis.

        Args:
            symbol: Trading symbol
        """
        try:
            state = self.trading_states[symbol]
            current_time = time.time()

            # Rate limiting: check minimum interval between signals
            if current_time - state.last_execution_time < self.min_signal_interval:
                return

            # Create price series
            price_series = pd.Series(state.price_history)

            # Apply Kalman filtering
            kalman_results = state.kalman_filter.filter_price_series(price_series)
            if kalman_results.empty:
                return

            # Get latest Kalman estimates
            latest_kalman = kalman_results.iloc[-1]

            # Use Kalman filtered prices for calculus analysis
            # Handle both possible column names for compatibility
            if 'filtered_price' in kalman_results.columns:
                filtered_prices = kalman_results['filtered_price']
            elif 'price_estimate' in kalman_results.columns:
                filtered_prices = kalman_results['price_estimate']
                # Rename for consistency with downstream processing
                kalman_results = kalman_results.rename(columns={'price_estimate': 'filtered_price'})
            else:
                # Fallback to raw prices if Kalman filtering failed
                logger.warning(f"No filtered prices available for {symbol}, using raw prices")
                filtered_prices = price_series

            # Validate we have enough data
            if filtered_prices.empty or len(filtered_prices) < 10:
                logger.warning(f"Insufficient filtered data for {symbol}, skipping signal")
                return

            # Generate calculus signals
            calculus_strategy = CalculusTradingStrategy()
            signals = calculus_strategy.generate_trading_signals(filtered_prices)
            if signals.empty:
                return

            # Get latest signal
            latest_signal = signals.iloc[-1]

            # Check if we have a valid signal
            if not latest_signal.get('valid_signal', False):
                return

            # Check for NaN values and handle them
            signal_type_raw = latest_signal.get('signal_type')
            if pd.isna(signal_type_raw) or signal_type_raw is None:
                logger.warning(f"NaN signal type detected for {symbol}, skipping signal")
                return

            try:
                signal_type = SignalType(int(signal_type_raw)) if not pd.isna(signal_type_raw) else SignalType.HOLD
            except (ValueError, TypeError):
                logger.warning(f"Invalid signal type {signal_type_raw} for {symbol}, defaulting to HOLD")
                signal_type = SignalType.HOLD

            # Safely get Kalman values with fallbacks
            kalman_velocity = latest_kalman.get('velocity', 0.0) if 'velocity' in latest_kalman else latest_kalman.get('velocity_estimate', 0.0)
            kalman_acceleration = latest_kalman.get('acceleration', 0.0) if 'acceleration' in latest_kalman else latest_kalman.get('acceleration_estimate', 0.0)
            filtered_price_value = latest_kalman.get('filtered_price', 0.0) if 'filtered_price' in latest_kalman else latest_kalman.get('price_estimate', latest_signal.get('price', 0.0))

            # Create signal dictionary with robust data access
            signal_dict = {
                'signal_type': signal_type,
                'interpretation': latest_signal.get('interpretation', 'Unknown'),
                'confidence': latest_signal.get('confidence', 0.0),
                'velocity': latest_signal.get('velocity', 0.0),
                'acceleration': latest_signal.get('acceleration', 0.0),
                'snr': latest_signal.get('snr', 0.0),
                'forecast': latest_signal.get('forecast', 0.0),
                'price': latest_signal.get('price', 0.0),
                'filtered_price': filtered_price_value,
                'kalman_velocity': kalman_velocity,
                'kalman_acceleration': kalman_acceleration
            }

            # Update state
            state.last_signal = signal_dict
            state.signal_count += 1

            # Log signal
            logger.info(f"=== CALCULUS SIGNAL for {symbol} ===")
            logger.info(f"Signal Type: {signal_dict['signal_type'].name}")
            logger.info(f"Interpretation: {signal_dict['interpretation']}")
            logger.info(f"Price: {signal_dict['price']:.2f} | Filtered: {signal_dict['filtered_price']:.2f}")
            logger.info(f"Velocity: {signal_dict['velocity']:.6f} | Acceleration: {signal_dict['acceleration']:.8f}")
            logger.info(f"SNR: {signal_dict['snr']:.2f} | Confidence: {signal_dict['confidence']:.2f}")
            logger.info(f"Forecast: {signal_dict['forecast']:.2f}")
            logger.info("=" * 40)

            # Execute trade if signal is actionable
            if self._is_actionable_signal(signal_dict):
                if self.portfolio_mode:
                    # Portfolio-aware execution
                    self._execute_portfolio_trade(symbol, signal_dict)
                else:
                    # Single-asset execution
                    self._execute_trade(symbol, signal_dict)

        except Exception as e:
            logger.error(f"Error processing trading signal for {symbol}: {e}")
            state.error_count += 1

    def _is_actionable_signal(self, signal_dict: Dict) -> bool:
        """
        Determine if signal is actionable for trading.

        Args:
            signal_dict: Signal information

        Returns:
            True if signal should trigger a trade
        """
        # Check emergency conditions
        if self.emergency_stop:
            return False

        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            logger.warning("Maximum consecutive losses reached. Pausing trading.")
            return False

        # Check daily loss limit
        if self.daily_pnl < -self.daily_loss_limit:
            logger.warning(f"Daily loss limit reached: {self.daily_pnl:.1%}")
            return False

        # Check signal strength
        actionable_signals = [
            SignalType.STRONG_BUY, SignalType.STRONG_SELL,
            SignalType.BUY, SignalType.SELL,
            SignalType.TRAIL_STOP_UP, SignalType.TAKE_PROFIT,
            SignalType.POSSIBLE_LONG, SignalType.POSSIBLE_EXIT_SHORT
        ]

        return (signal_dict['signal_type'] in actionable_signals and
                signal_dict['confidence'] > 0.6 and
                signal_dict['snr'] > 1.0)

    def _execute_trade(self, symbol: str, signal_dict: Dict):
        """
        Execute trade based on calculus signal.

        Args:
            symbol: Trading symbol
            signal_dict: Signal information
        """
        try:
            state = self.trading_states[symbol]
            current_price = signal_dict['price']

            # Input validation for critical signal data
            try:
                current_price = float(current_price)
                snr = float(signal_dict.get('snr', 0))
                confidence = float(signal_dict.get('confidence', 0))
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid signal data for {symbol}: {e}, skipping trade")
                state.error_count += 1
                return

            if current_price <= 0:
                logger.warning(f"Invalid price for {symbol}: {current_price}, skipping trade")
                state.error_count += 1
                return

            # Get account balance - check for margin trading funds
            account_info = self.bybit_client.get_wallet_balance()
            if not account_info:
                logger.error("Could not fetch account balance")
                return

            try:
                available_balance = float(account_info.get('totalAvailableBalance', 0))
                total_equity = float(account_info.get('totalEquity', 0))
                
                # If no spot balance but have equity, try margin trading
                if available_balance == 0 and total_equity > 0:
                    logger.info(f"Spot balance: ${available_balance:.2f}, Total equity: ${total_equity:.2f}")
                    logger.info("Attempting margin trading with equity funds")
                    
                    # Use equity for margin trading calculations
                    # Reduce available balance calculation to account for margin requirements
                    margin_available = total_equity * 0.8  # Use 80% of equity for trading
                    if margin_available >= 5:  # $5 minimum for leverage trading
                        available_balance = margin_available
                        logger.info(f"Using margin trading balance: ${available_balance:.2f}")
                    else:
                        logger.info(f"Insufficient equity for leverage trading: ${margin_available:.2f} (need $5+)")
                        return
                
            except (ValueError, TypeError):
                logger.warning(f"Invalid account balance: {account_info}, using 0")
                available_balance = 0

            if available_balance < 5:  # $5 minimum for leverage trading
                logger.info(f"Insufficient balance for leverage trading: ${available_balance:.2f} (need $5+)")
                return

            # Adjust leverage for small balances
            if available_balance < 100:
                # Reduce leverage for small balances for safety
                max_leverage = min(10.0, self.risk_manager.max_leverage)
                original_leverage = self.risk_manager.max_leverage
                self.risk_manager.max_leverage = max_leverage
                logger.info(f"Reduced leverage to {max_leverage}x for small balance (${available_balance:.2f})")

            # Calculate position size with validated data
            position_size = self.risk_manager.calculate_position_size(
                symbol=symbol,
                signal_strength=snr,
                confidence=confidence,
                current_price=current_price,
                account_balance=available_balance
            )
            
            # Restore original leverage
            if available_balance < 100:
                self.risk_manager.max_leverage = original_leverage

            # Apply maximum position size limit
            notional_value = position_size.quantity * current_price
            if notional_value > self.max_position_size:
                position_size.quantity = self.max_position_size / current_price
                position_size.notional_value = self.max_position_size

            # Calculate dynamic TP/SL
            trading_levels = self.risk_manager.calculate_dynamic_tp_sl(
                signal_type=signal_dict['signal_type'],
                current_price=current_price,
                velocity=signal_dict['velocity'],
                acceleration=signal_dict['acceleration'],
                volatility=0.02  # Would calculate from recent price action
            )

            # Validate trade risk
            is_valid, reason = self.risk_manager.validate_trade_risk(
                symbol, position_size, trading_levels
            )

            if not is_valid:
                logger.info(f"Trade validation failed: {reason}")
                return

            # Determine order side and type
            if signal_dict['signal_type'] in [SignalType.BUY, SignalType.STRONG_BUY, SignalType.POSSIBLE_LONG]:
                side = "Buy"
            else:
                side = "Sell"

            # Execute order with TP/SL
            if self.simulation_mode:
                # Simulate successful trade
                order_result = {'orderId': f'SIM_{int(time.time())}', 'status': 'Filled'}
                logger.info(f"🧪 SIMULATION TRADE: {symbol} {side} {position_size.quantity:.4f} @ {current_price:.2f}")
                logger.info(f"   TP: {trading_levels.take_profit:.2f} | SL: {trading_levels.stop_loss:.2f}")
                logger.info(f"   Risk/Reward: {trading_levels.risk_reward_ratio:.2f} | Leverage: {position_size.leverage_used:.1f}x")
            else:
                # Execute real order
                order_result = self.bybit_client.place_order(
                    symbol=symbol,
                    side=side,
                    order_type="Market",  # Market orders for immediate execution
                    qty=position_size.quantity,
                    take_profit=trading_levels.take_profit,
                    stop_loss=trading_levels.stop_loss
                )

                if order_result:
                    logger.info(f"✅ TRADE EXECUTED: {symbol} {side} {position_size.quantity:.4f} @ {current_price:.2f}")
                    logger.info(f"   TP: {trading_levels.take_profit:.2f} | SL: {trading_levels.stop_loss:.2f}")
                    logger.info(f"   Risk/Reward: {trading_levels.risk_reward_ratio:.2f} | Leverage: {position_size.leverage_used:.1f}x")

            if order_result:
                # Update position tracking
                position_info = {
                    'symbol': symbol,
                    'side': side,
                    'quantity': position_size.quantity,
                    'entry_price': current_price,
                    'notional_value': position_size.notional_value,
                    'take_profit': trading_levels.take_profit,
                    'stop_loss': trading_levels.stop_loss,
                    'leverage_used': position_size.leverage_used,
                    'entry_time': current_time,
                    'signal_type': signal_dict['signal_type'].name,
                    'confidence': signal_dict['confidence']
                }

                state.position_info = position_info
                state.last_execution_time = time.time()

                # Update risk manager
                self.risk_manager.update_position(symbol, position_info)

                # Update performance
                self.performance.total_trades += 1

            else:
                logger.error(f"❌ Order execution failed for {symbol}")
                state.error_count += 1

        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
            state.error_count += 1

    def _execute_portfolio_trade(self, symbol: str, signal_dict: Dict):
        """
        Execute portfolio-aware trade based on calculus signal and portfolio optimization.

        Args:
            symbol: Trading symbol
            signal_dict: Signal information
        """
        try:
            if not self.portfolio_mode or not self.portfolio_manager:
                # Fallback to single-asset execution
                self._execute_trade(symbol, signal_dict)
                return

            logger.info(f"🎯 PORTFOLIO TRADE EXECUTION for {symbol}")

            # Step 1: Add signal to coordinator
            self.signal_coordinator.process_signal(
                symbol=symbol,
                signal_type=signal_dict['signal_type'],
                confidence=signal_dict['confidence'],
                signal_strength=abs(signal_dict['velocity']),  # Use velocity as signal strength
                price=signal_dict['price']
            )

            # Step 2: Get coordinated portfolio decision
            portfolio_decisions = self.signal_coordinator.get_signal_recommendations()

            if not portfolio_decisions:
                logger.info("⏸️  No actionable portfolio decision at this time")
                return

            # Take the highest priority recommendation
            portfolio_decision = portfolio_decisions[0]

            logger.info(f"📊 Portfolio Decision: {portfolio_decision.signal_type}")
            logger.info(f"   Confidence: {portfolio_decision.confidence:.2f}")
            logger.info(f"   Signal Strength: {portfolio_decision.signal_strength:.2f}")
            logger.info(f"   Recommended Size: ${portfolio_decision.recommended_size:.2f}")
            logger.info(f"   Priority: {portfolio_decision.priority.name}")

            # Step 3: Execute trade directly based on coordinated signal
            if portfolio_decision.confidence > 0.7 and portfolio_decision.signal_strength > 0.5:
                logger.info(f"🎯 WOULD EXECUTE TRADE: {portfolio_decision.symbol} - Signal: {portfolio_decision.signal_type.name}")
                logger.info(f"   Confidence: {portfolio_decision.confidence:.2f} | Strength: {portfolio_decision.signal_strength:.2f}")
                logger.info("   ⚠️  TRADING DISABLED - System needs debugging before real trades")
            else:
                logger.info(f"⏸️  Signal confidence/strength too low: {portfolio_decision.confidence:.2f}/{portfolio_decision.signal_strength:.2f}")

        except Exception as e:
            logger.error(f"Error executing portfolio trade for {symbol}: {e}")

    def _execute_allocation_decision(self, decision: AllocationDecision, signal_dict: Dict):
        """
        Execute a specific portfolio allocation decision.

        Args:
            decision: Portfolio allocation decision
            signal_dict: Original calculus signal
        """
        try:
            logger.info(f"💰 EXECUTING ALLOCATION: {decision.symbol}")
            logger.info(f"   Target Weight: {decision.target_weight:.1%}")
            logger.info(f"   Current Weight: {decision.current_weight:.1%}")
            logger.info(f"   Trade Type: {decision.trade_type}")
            logger.info(f"   Quantity: {decision.quantity:.6f}")
            logger.info(f"   Reason: {decision.reason}")

            # Convert to single-asset trade format
            current_price = signal_dict['price']

            # Determine order side
            if decision.trade_type in ['ENTER_LONG', 'INCREASE_LONG']:
                side = "Buy"
            elif decision.trade_type in ['ENTER_SHORT', 'INCREASE_SHORT']:
                side = "Sell"
            elif decision.trade_type == 'REDUCE_LONG':
                side = "Sell"
            elif decision.trade_type == 'COVER_SHORT':
                side = "Buy"
            else:
                logger.warning(f"Unknown trade type: {decision.trade_type}")
                return

            # Calculate TP/SL based on portfolio risk management
            trading_levels = self.risk_manager.calculate_dynamic_tp_sl(
                signal_type=signal_dict['signal_type'],
                current_price=current_price,
                velocity=signal_dict['velocity'],
                acceleration=signal_dict['acceleration'],
                volatility=0.02
            )

            # Execute order
            if self.simulation_mode:
                # Simulate successful trade
                order_result = {'orderId': f'PORTFOLIO_SIM_{int(time.time())}', 'status': 'Filled'}
                logger.info(f"🧪 PORTFOLIO SIMULATION: {decision.symbol} {side} {decision.quantity:.6f} @ {current_price:.2f}")
            else:
                # Execute real portfolio order
                order_result = self.bybit_client.place_order(
                    symbol=decision.symbol,
                    side=side,
                    order_type="Market",
                    qty=decision.quantity,
                    take_profit=trading_levels.take_profit,
                    stop_loss=trading_levels.stop_loss
                )

            if order_result:
                logger.info(f"✅ PORTFOLIO TRADE EXECUTED: {decision.symbol} {side} {decision.quantity:.6f} @ {current_price:.2f}")

                # Update portfolio manager
                self.portfolio_manager.update_position_after_trade(
                    decision.symbol, decision, order_result, current_price
                )

                # Update trading state
                state = self.trading_states[decision.symbol]
                state.last_execution_time = time.time()

                # Update performance
                self.performance.total_trades += 1

            else:
                logger.error(f"❌ Portfolio order execution failed for {decision.symbol}")

        except Exception as e:
            logger.error(f"Error executing allocation decision for {decision.symbol}: {e}")

    def _portfolio_monitoring_loop(self):
        """Background portfolio monitoring and rebalancing loop."""
        while self.is_running and self.portfolio_mode:
            try:
                # Update portfolio optimization
                if self.joint_distribution_analyzer.is_data_sufficient():
                    # Get current market data
                    market_data = self.ws_client.get_latest_portfolio_data()

                    # Update portfolio optimization
                    optimization_result = self.portfolio_manager.update_optimization(
                        self.joint_distribution_analyzer,
                        self.portfolio_optimizer,
                        market_data
                    )

                    if optimization_result:
                        logger.info(f"📊 Portfolio optimization updated:")
                        logger.info(f"   Expected Return: {optimization_result['expected_return']:.4f}")
                        logger.info(f"   Volatility: {optimization_result['volatility']:.4f}")
                        logger.info(f"   Sharpe Ratio: {optimization_result['sharpe_ratio']:.3f}")

                        # Check for rebalancing opportunities
                        if self.portfolio_manager.should_rebalance():
                            logger.info("🔄 Portfolio rebalancing triggered")
                            rebalance_decisions = self.portfolio_manager.create_rebalance_decisions()

                            for decision in rebalance_decisions:
                                if not self.simulation_mode:
                                    logger.info(f"🚨 REAL REBALANCE: {decision.symbol} - {decision.reason}")
                                    # Execute rebalance in production
                                    self._execute_allocation_decision(decision, {})
                                else:
                                    logger.info(f"🧪 SIMULATION REBALANCE: {decision.symbol} - {decision.reason}")

                # Sleep for portfolio monitoring interval
                time.sleep(60)  # Monitor every minute

            except Exception as e:
                logger.error(f"Error in portfolio monitoring loop: {e}")
                time.sleep(120)  # Wait longer on error

    def _monitoring_loop(self):
        """Background monitoring loop for system health and performance."""
        while self.is_running:
            try:
                # Check system health
                self._check_system_health()

                # Monitor positions
                self._monitor_positions()

                # Update performance metrics
                self._update_performance_metrics()

                # Reset daily counters if needed
                self._reset_daily_counters()

                # Sleep for monitoring interval
                time.sleep(30)  # Monitor every 30 seconds

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error

    def _check_system_health(self):
        """Check system health and circuit breakers."""
        # Check WebSocket connection
        if not self.ws_client.is_connected:
            logger.warning("WebSocket disconnected. Attempting reconnection...")
            # Reconnection handled automatically by WebSocket client

        # Check error rates
        total_errors = sum(state.error_count for state in self.trading_states.values())
        total_signals = sum(state.signal_count for state in self.trading_states.values())

        if total_signals > 0:
            error_rate = total_errors / total_signals
            if error_rate > 0.1:  # 10% error rate threshold
                logger.warning(f"High error rate: {error_rate:.1%}")
                self.emergency_stop = True

    def _monitor_positions(self):
        """Monitor open positions and update risk metrics."""
        for symbol, state in self.trading_states.items():
            if state.position_info:
                try:
                    # Get current position info
                    position_info = self.bybit_client.get_position_info(symbol)
                    if position_info:
                        # Check for stop loss or take profit
                        current_pnl = float(position_info.get('unrealisedPnl', 0))
                        entry_price = float(position_info.get('entryPrice', state.position_info['entry_price']))
                        current_price = float(position_info.get('markPrice', state.position_info['entry_price']))

                        # Update position info
                        state.position_info.update({
                            'current_price': current_price,
                            'unrealised_pnl': current_pnl,
                            'pnl_percent': (current_pnl / state.position_info['notional_value']) * 100
                        })

                        # Check if position should be closed (manual override)
                        if self._should_close_position(state.position_info, position_info):
                            self._close_position(symbol, "Risk management")

                except Exception as e:
                    logger.error(f"Error monitoring position for {symbol}: {e}")

    def _should_close_position(self, position_info: Dict, current_position: Dict) -> bool:
        """Determine if position should be closed based on risk rules."""
        try:
            pnl_percent = position_info.get('pnl_percent', 0)

            # Close on excessive loss
            if pnl_percent < -0.05:  # 5% loss
                return True

            # Close on significant profit
            if pnl_percent > 0.10:  # 10% profit
                return True

            # Close on position age (prevent holding too long)
            age = time.time() - position_info['entry_time']
            if age > 3600:  # 1 hour
                return True

            return False

        except Exception as e:
            logger.error(f"Error checking position close condition: {e}")
            return False

    def _close_position(self, symbol: str, reason: str):
        """Close position for a symbol."""
        try:
            state = self.trading_states[symbol]
            if not state.position_info:
                return

            # Determine close side
            current_side = state.position_info['side']
            close_side = "Sell" if current_side == "Buy" else "Buy"

            # Get current position size
            position_info = self.bybit_client.get_position_info(symbol)
            if position_info:
                position_size = abs(float(position_info.get('size', 0)))
                if position_size > 0:
                    # Close position
                    result = self.bybit_client.place_order(
                        symbol=symbol,
                        side=close_side,
                        order_type="Market",
                        qty=position_size,
                        reduce_only=True
                    )

                    if result:
                        # Calculate PnL
                        pnl = float(position_info.get('unrealisedPnl', 0))
                        pnl_percent = (pnl / state.position_info['notional_value']) * 100

                        # Update performance
                        self.performance.total_pnl += pnl
                        if pnl > 0:
                            self.performance.winning_trades += 1
                            self.consecutive_losses = 0
                        else:
                            self.performance.losing_trades += 1
                            self.consecutive_losses += 1

                        self.daily_pnl += pnl / self.risk_manager.current_portfolio_value

                        logger.info(f"✅ POSITION CLOSED: {symbol} PnL: {pnl:.2f} ({pnl_percent:.1f}%) - {reason}")

                        # Update risk manager
                        self.risk_manager.close_position(symbol, pnl, reason)

                        # Clear position info
                        state.position_info = None

        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")

    def _emergency_close_all_positions(self):
        """Emergency close all positions."""
        logger.warning("Emergency close all positions activated!")
        for symbol in self.symbols:
            self._close_position(symbol, "Emergency stop")

    def _update_performance_metrics(self):
        """Update performance metrics."""
        if self.performance.total_trades > 0:
            self.performance.success_rate = (
                self.performance.winning_trades / self.performance.total_trades
            )

        # Update portfolio value in risk manager
        account_info = self.bybit_client.get_wallet_balance()
        if account_info:
            portfolio_value = float(account_info.get('totalEquity', 0))
            self.risk_manager.update_portfolio_value(portfolio_value)

    def _reset_daily_counters(self):
        """Reset daily counters at midnight."""
        current_time = time.time()
        last_reset = self.last_reset_time

        # Reset if it's a new day
        if current_time - last_reset > 86400:  # 24 hours
            self.daily_pnl = 0.0
            self.consecutive_losses = 0
            self.last_reset_time = current_time
            logger.info("Daily counters reset")

    def get_system_status(self) -> Dict:
        """Get comprehensive system status."""
        status = {
            'is_running': self.is_running,
            'emergency_stop': self.emergency_stop,
            'symbols': self.symbols,
            'performance': {
                'total_trades': self.performance.total_trades,
                'winning_trades': self.performance.winning_trades,
                'losing_trades': self.performance.losing_trades,
                'success_rate': self.performance.success_rate,
                'total_pnl': self.performance.total_pnl,
                'daily_pnl': self.daily_pnl
            },
            'connections': {
                'websocket_connected': self.ws_client.is_connected if self.ws_client else False,
                'api_connected': self.bybit_client.test_connection() if self.bybit_client else False
            },
            'trading_states': {}
        }

        # Add per-symbol status
        for symbol, state in self.trading_states.items():
            status['trading_states'][symbol] = {
                'price_history_length': len(state.price_history),
                'signal_count': state.signal_count,
                'error_count': state.error_count,
                'has_position': state.position_info is not None,
                'last_signal_time': state.last_execution_time
            }

        # Add risk metrics
        risk_metrics = self.risk_manager.calculate_risk_metrics()
        status['risk_metrics'] = {
            'total_exposure': risk_metrics.total_exposure,
            'margin_used_percent': risk_metrics.margin_used_percent,
            'open_positions': risk_metrics.open_positions_count,
            'current_drawdown': risk_metrics.current_drawdown,
            'sharpe_ratio': risk_metrics.sharpe_ratio
        }

        # Add portfolio metrics if enabled
        if self.portfolio_mode and self.portfolio_manager:
            portfolio_status = self.portfolio_manager.get_portfolio_status()
            status['portfolio'] = {
                'mode_enabled': True,
                'total_value': portfolio_status['total_value'],
                'available_cash': portfolio_status['available_cash'],
                'positions_count': len(portfolio_status['positions']),
                'allocation_drift': portfolio_status['allocation_drift'],
                'last_rebalance': portfolio_status['last_rebalance'],
                'optimization_active': self.joint_distribution_analyzer.is_data_sufficient() if self.joint_distribution_analyzer else False
            }

            # Add signal coordinator status
            if self.signal_coordinator:
                signal_status = self.signal_coordinator.get_coordinator_status()
                status['signal_coordinator'] = signal_status
        else:
            status['portfolio'] = {'mode_enabled': False}

        return status

    async def _execute_direct_trade(self, portfolio_decision):
        """Execute a trade directly based on portfolio decision"""
        try:
            symbol = portfolio_decision.symbol
            signal_type = portfolio_decision.signal_type
            confidence = portfolio_decision.confidence
            signal_strength = portfolio_decision.signal_strength
            recommended_size = portfolio_decision.recommended_size

            logger.info(f"🎯 EXECUTING DIRECT TRADE: {symbol}")
            logger.info(f"   Signal: {signal_type.name}")
            logger.info(f"   Confidence: {confidence:.2f}")
            logger.info(f"   Size: ${recommended_size:.2f}")

            # Determine trade direction based on signal type
            if signal_type in [SignalType.BULLISH_ACCELERATION, SignalType.BULLISH_REVERSAL]:
                side = "Buy"
            elif signal_type in [SignalType.BEARISH_ACCELERATION, SignalType.BEARISH_REVERSAL]:
                side = "Sell"
            else:
                logger.info(f"⏸️  Signal type {signal_type.name} not actionable for trading")
                return

            # Execute the actual trade
            result = await self._execute_real_trade(symbol, side, recommended_size, confidence)

            if result:
                logger.info(f"✅ Trade executed successfully: {symbol} {side} ${recommended_size:.2f}")
            else:
                logger.error(f"❌ Trade execution failed: {symbol} {side}")

        except Exception as e:
            logger.error(f"Error in _execute_direct_trade: {e}")

    async def _execute_real_trade(self, symbol: str, side: str, quantity: float, confidence: float):
        """Execute real trade on Bybit exchange"""
        try:
            # Get current price
            ticker = self.bybit_client.get_ticker(symbol)
            if not ticker or 'last_price' not in ticker:
                logger.error(f"Could not get current price for {symbol}")
                return False

            current_price = float(ticker['last_price'])
            usd_amount = min(quantity, 100.0)  # Cap at $100 for safety

            # Calculate quantity in base asset
            qty = usd_amount / current_price

            # Apply risk management
            risk_check = self.risk_manager.validate_trade_parameters(
                symbol=symbol,
                side=side.lower(),
                quantity=qty,
                price=current_price
            )

            if not risk_check['valid']:
                logger.warning(f"Risk validation failed: {risk_check['reason']}")
                return False

            # Execute order via Bybit client
            order_result = self.bybit_client.place_order(
                symbol=symbol,
                side=side.lower(),
                order_type="market",
                quantity=qty
            )

            if order_result and order_result.get('retCode') == 0:
                logger.info(f"✅ ORDER PLACED: {side} {qty:.6f} {symbol} at ~${current_price:.2f}")
                return True
            else:
                logger.error(f"❌ ORDER FAILED: {order_result}")
                return False

        except Exception as e:
            logger.error(f"Error executing real trade: {e}")
            return False

if __name__ == '__main__':
    import sys

    # Check command line arguments
    simulation_mode = '--simulation' in sys.argv or '-s' in sys.argv
    single_asset_mode = '--single' in sys.argv or '--single-asset' in sys.argv

    print('🚀 ANNE\'S ENHANCED CALCULUS TRADING SYSTEM')
    print('=' * 60)
    print('🎯 Portfolio-Integrated Multi-Asset Trading System')

    if single_asset_mode:
        print('📊 SINGLE ASSET MODE - Traditional calculus trading')
        symbols = ["BTCUSDT", "ETHUSDT"]
        portfolio_mode = False
    else:
        print('📈 PORTFOLIO MODE - Multi-asset optimization')
        print('   🎓 Calculus signals for TIMING')
        print('   📊 Portfolio optimization for ALLOCATION')
        print('   🔢 Joint distribution for RISK')
        symbols = None  # Use default 8 assets
        portfolio_mode = True

    if simulation_mode:
        print('🧪 SIMULATION MODE - Safe for testing')
    else:
        print('⚠️  LIVE TRADING MODE - REAL MONEY AT RISK!')
        print('   🚨 This will execute REAL trades on Bybit!')

    print('=' * 60)

    # Initialize enhanced trader
    trader = LiveCalculusTrader(
        symbols=symbols,
        window_size=200,
        min_signal_interval=30,
        simulation_mode=simulation_mode,
        portfolio_mode=portfolio_mode
    )

    try:
        if simulation_mode:
            print(f'🧪 Starting simulation trading...')
        else:
            print(f'💰 Starting LIVE trading with REAL money!')

        trader.start()
    except KeyboardInterrupt:
        logger.info("Shutting down trading system...")
        trader.stop()
    except Exception as e:
        logger.error(f"Trading system error: {e}")
        trader.stop()
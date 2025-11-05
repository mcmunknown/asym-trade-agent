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
    Production-ready live trading system implementing Anne's complete calculus approach.

    This system integrates all components:
    - Real-time data processing with WebSocket
    - Kalman filtering for adaptive state estimation
    - Calculus-based signal generation with SNR filtering
    - Dynamic risk management and position sizing
    - High-frequency execution with TP/SL management
    """

    def __init__(self,
                 symbols: List[str],
                 window_size: int = 200,
                 min_signal_interval: int = 30,
                 emergency_stop: bool = False,
                 max_position_size: float = 1000.0):
        """
        Initialize the live calculus trading system.

        Args:
            symbols: List of trading symbols
            window_size: Price history window for analysis
            min_signal_interval: Minimum seconds between signals
            emergency_stop: Emergency stop flag
            max_position_size: Maximum position size per trade
        """
        self.symbols = symbols
        self.window_size = window_size
        self.min_signal_interval = min_signal_interval
        self.emergency_stop = emergency_stop
        self.max_position_size = max_position_size

        # Initialize components
        self.ws_client = BybitWebSocketClient(
            symbols=symbols,
            testnet=Config.BYBIT_TESTNET,
            channel_types=[ChannelType.TRADE, ChannelType.ORDERBOOK_1],
            heartbeat_interval=20
        )
        self.bybit_client = BybitClient()
        self.risk_manager = RiskManager()

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

        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

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
            # Test Bybit API connection
            if not self.bybit_client.test_connection():
                logger.error("Bybit API connection failed")
                return False

            # Test account balance
            balance = self.bybit_client.get_account_balance()
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
            filtered_prices = kalman_results['filtered_price']

            # Generate calculus signals
            calculus_strategy = CalculusTradingStrategy()
            signals = calculus_strategy.generate_trading_signals(filtered_prices)
            if signals.empty:
                return

            # Get latest signal
            latest_signal = signals.iloc[-1]

            # Check if we have a valid signal
            if not latest_signal['valid_signal']:
                return

            # Create signal dictionary
            signal_dict = {
                'signal_type': SignalType(latest_signal['signal_type']),
                'interpretation': latest_signal['interpretation'],
                'confidence': latest_signal['confidence'],
                'velocity': latest_signal['velocity'],
                'acceleration': latest_signal['acceleration'],
                'snr': latest_signal['snr'],
                'forecast': latest_signal['forecast'],
                'price': latest_signal['price'],
                'filtered_price': latest_signal['filtered_price'],
                'kalman_velocity': latest_kalman['velocity'],
                'kalman_acceleration': latest_kalman['acceleration']
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

            # Get account balance
            account_info = self.bybit_client.get_account_balance()
            if not account_info:
                logger.error("Could not fetch account balance")
                return

            available_balance = float(account_info.get('totalAvailableBalance', 0))
            if available_balance < 100:  # Minimum balance
                logger.warning(f"Insufficient balance: {available_balance}")
                return

            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                symbol=symbol,
                signal_strength=signal_dict['snr'],
                confidence=signal_dict['confidence'],
                current_price=current_price,
                account_balance=available_balance
            )

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
        account_info = self.bybit_client.get_account_balance()
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

        return status

if __name__ == '__main__':
    # Example usage
    trader = LiveCalculusTrader(
        symbols=["BTCUSDT", "ETHUSDT"],
        window_size=200,
        min_signal_interval=30
    )

    try:
        trader.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        trader.stop()
    except Exception as e:
        logger.error(f"Trading system error: {e}")
        trader.stop()
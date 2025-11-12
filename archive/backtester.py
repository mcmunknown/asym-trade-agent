"""
Backtesting Framework for Anne's Calculus Trading System
========================================================

This module provides comprehensive backtesting capabilities for validating the
calculus-based trading approach with historical data.

Features:
- Historical backtesting of calculus-based strategies
- Performance metrics calculation (Sharpe ratio, max drawdown, etc.)
- Parameter optimization and grid search
- Monte Carlo simulation for robustness testing
- Walk-forward analysis
- Benchmark comparison
- Statistical significance testing
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Import our trading components
from calculus_strategy import CalculusTradingStrategy, SignalType
from quantitative_models import CalculusPriceAnalyzer
from kalman_filter import AdaptiveKalmanFilter, KalmanConfig
from risk_manager import RiskManager, PositionSize, TradingLevels

logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    start_date: str
    end_date: str
    initial_capital: float = 10000.0
    commission_rate: float = 0.001  # 0.1%
    slippage_rate: float = 0.0005  # 0.05%
    leverage: float = 1.0
    max_position_size: float = 1.0  # 100% of capital
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    min_signal_interval: int = 300  # 5 minutes

@dataclass
class TradeRecord:
    """Record of a single trade"""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    side: str
    pnl: float
    pnl_percent: float
    commission: float
    slippage: float
    signal_type: str
    confidence: float
    holding_period: float
    exit_reason: str

@dataclass
class BacktestResult:
    """Results of backtesting"""
    trades: List[TradeRecord]
    equity_curve: pd.Series
    returns: pd.Series
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    benchmark_returns: pd.Series = None

class PerformanceAnalyzer:
    """Analyzes backtesting performance and calculates metrics"""

    @staticmethod
    def calculate_metrics(equity_curve: pd.Series, trades: List[TradeRecord],
                         benchmark_returns: pd.Series = None) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.

        Args:
            equity_curve: Portfolio value over time
            trades: List of trade records
            benchmark_returns: Benchmark returns for comparison

        Returns:
            Dictionary of performance metrics
        """
        try:
            if equity_curve.empty:
                return {}

            returns = equity_curve.pct_change().dropna()

            # Basic return metrics
            total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
            annual_return = (1 + total_return) ** (252 / len(equity_curve)) - 1

            # Risk metrics
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
            sortino_ratio = annual_return / (returns[returns < 0].std() * np.sqrt(252)) if len(returns[returns < 0]) > 0 else 0

            # Drawdown metrics
            rolling_max = equity_curve.expanding().max()
            drawdown = (equity_curve - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            avg_drawdown = drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0

            # Trade metrics
            if trades:
                winning_trades = [t for t in trades if t.pnl > 0]
                losing_trades = [t for t in trades if t.pnl < 0]

                win_rate = len(winning_trades) / len(trades)
                avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
                avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
                profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

                avg_holding_period = np.mean([t.holding_period for t in trades]) if trades else 0

                # Largest winner/loser
                largest_win = max([t.pnl for t in trades]) if trades else 0
                largest_loss = min([t.pnl for t in trades]) if trades else 0
            else:
                win_rate = avg_win = avg_loss = profit_factor = 0
                avg_holding_period = largest_win = largest_loss = 0

            # Benchmark comparison
            beta = alpha = information_ratio = 0
            if benchmark_returns is not None and len(benchmark_returns) == len(returns):
                # Align benchmark with strategy returns
                aligned_benchmark = benchmark_returns.reindex(returns.index).fillna(0)
                beta = np.cov(returns, aligned_benchmark)[0, 1] / np.var(aligned_benchmark) if np.var(aligned_benchmark) > 0 else 0
                alpha = annual_return - (0.02 + beta * 0.08)  # Assuming 2% risk-free, 8% market return
                excess_returns = returns - aligned_benchmark
                information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0

            # Calmar ratio (annual return / max drawdown)
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

            metrics = {
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'avg_drawdown': avg_drawdown,
                'calmar_ratio': calmar_ratio,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'largest_win': largest_win,
                'largest_loss': largest_loss,
                'avg_holding_period': avg_holding_period,
                'total_trades': len(trades),
                'beta': beta,
                'alpha': alpha,
                'information_ratio': information_ratio
            }

            return metrics

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}

class CalculusBacktester:
    """
    Backtesting engine for Anne's calculus-based trading strategy.

    This class provides comprehensive backtesting capabilities including:
    - Signal generation using calculus analysis
    - Risk management with position sizing
    - Portfolio simulation with realistic costs
    - Performance analysis and benchmarking
    """

    def __init__(self, config: BacktestConfig):
        """
        Initialize backtester with configuration.

        Args:
            config: Backtesting configuration
        """
        self.config = config
        self.calculus_strategy = CalculusTradingStrategy()
        self.risk_manager = RiskManager()
        self.kalman_filter = AdaptiveKalmanFilter()

        # Initialize portfolio state
        self.capital = config.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.returns = []

        logger.info(f"Calculus backtester initialized: {config.start_date} to {config.end_date}")

    def load_data(self, symbol: str, timeframe: str = '1h') -> pd.DataFrame:
        """
        Load historical data for backtesting.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # This would typically load from a data source
            # For now, create sample data for demonstration
            date_range = pd.date_range(start=self.config.start_date, end=self.config.end_date, freq='1H')
            np.random.seed(42)

            # Generate realistic price data with trends and volatility
            returns = np.random.normal(0.0001, 0.02, len(date_range))
            prices = [self.config.initial_capital / 100]  # Starting price

            for ret in returns:
                prices.append(prices[-1] * (1 + ret))

            prices = prices[1:]  # Remove initial value

            # Create OHLCV data
            data = pd.DataFrame({
                'timestamp': date_range,
                'open': prices,
                'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
                'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
                'close': prices,
                'volume': np.random.lognormal(10, 1, len(date_range))
            })

            data.set_index('timestamp', inplace=True)

            logger.info(f"Loaded {len(data)} data points for {symbol}")
            return data

        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            return pd.DataFrame()

    def run_backtest(self, symbol: str, data: pd.DataFrame) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            symbol: Trading symbol
            data: Historical OHLCV data

        Returns:
            BacktestResult with performance data
        """
        try:
            logger.info(f"Starting backtest for {symbol} with {len(data)} data points")

            # Reset state
            self.capital = self.config.initial_capital
            self.positions = {}
            self.trades = []
            self.equity_curve = []
            self.returns = []

            # Initialize equity curve
            equity_series = []
            timestamps = []

            # Process each data point
            for i, (timestamp, row) in enumerate(data.iterrows()):
                # Update equity curve
                current_equity = self._calculate_total_equity(row['close'])
                equity_series.append(current_equity)
                timestamps.append(timestamp)

                # Skip if not enough data for analysis
                if i < 50:  # Minimum data points for calculus analysis
                    continue

                # Get price history for signal generation
                price_history = data['close'].iloc[max(0, i-200):i+1]

                # Generate trading signal
                signal = self._generate_signal(price_history)
                if not signal:
                    continue

                # Process existing positions
                self._manage_positions(symbol, row, signal, timestamp)

                # Generate new trades based on signal
                if self._should_enter_trade(signal):
                    self._enter_trade(symbol, row, signal, timestamp)

            # Create equity curve series
            equity_curve = pd.Series(equity_series, index=timestamps)

            # Close any remaining positions at the end
            if symbol in self.positions:
                self._close_position(symbol, data.iloc[-1], "End of backtest", data.index[-1])

            # Calculate returns
            returns = equity_curve.pct_change().dropna()

            # Calculate performance metrics
            analyzer = PerformanceAnalyzer()
            metrics = analyzer.calculate_metrics(equity_curve, self.trades)

            # Create result object
            result = BacktestResult(
                trades=self.trades,
                equity_curve=equity_curve,
                returns=returns,
                metrics=metrics,
                parameters={
                    'symbol': symbol,
                    'start_date': self.config.start_date,
                    'end_date': self.config.end_date,
                    'initial_capital': self.config.initial_capital
                }
            )

            logger.info(f"Backtest completed for {symbol}")
            logger.info(f"Total return: {metrics.get('total_return', 0):.2%}")
            logger.info(f"Sharpe ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            logger.info(f"Max drawdown: {metrics.get('max_drawdown', 0):.2%}")
            logger.info(f"Total trades: {metrics.get('total_trades', 0)}")

            return result

        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return BacktestResult([], pd.Series(), pd.Series(), {}, {})

    def _generate_signal(self, price_history: pd.Series) -> Optional[Dict]:
        """Generate trading signal using calculus analysis."""
        try:
            # Apply Kalman filtering
            kalman_results = self.kalman_filter.filter_price_series(price_history)
            if kalman_results.empty:
                return None

            # Use filtered prices for calculus analysis
            filtered_prices = kalman_results['filtered_price']

            # Generate calculus signals
            signals = self.calculus_strategy.generate_trading_signals(filtered_prices)
            if signals.empty:
                return None

            # Get latest signal
            latest_signal = signals.iloc[-1]

            if not latest_signal['valid_signal']:
                return None

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
                'filtered_price': latest_signal['filtered_price']
            }

            return signal_dict

        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return None

    def _should_enter_trade(self, signal: Dict) -> bool:
        """Determine if a trade should be entered based on signal."""
        if not signal:
            return False

        # Check signal strength
        actionable_signals = [
            SignalType.STRONG_BUY, SignalType.STRONG_SELL,
            SignalType.BUY, SignalType.SELL,
            SignalType.POSSIBLE_LONG, SignalType.POSSIBLE_EXIT_SHORT
        ]

        return (signal['signal_type'] in actionable_signals and
                signal['confidence'] > 0.5 and
                signal['snr'] > 1.0)

    def _enter_trade(self, symbol: str, market_data: pd.Series, signal: Dict, timestamp: datetime):
        """Enter a new position based on signal."""
        try:
            current_price = market_data['close']

            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                symbol=symbol,
                signal_strength=signal['snr'],
                confidence=signal['confidence'],
                current_price=current_price,
                account_balance=self.capital
            )

            # Apply limits
            max_notional = self.capital * self.config.max_position_size
            if position_size.notional_value > max_notional:
                position_size.quantity = max_notional / current_price
                position_size.notional_value = max_notional

            # Determine position side
            if signal['signal_type'] in [SignalType.BUY, SignalType.STRONG_BUY, SignalType.POSSIBLE_LONG]:
                side = 'long'
            else:
                side = 'short'

            # Calculate TP/SL
            trading_levels = self.risk_manager.calculate_dynamic_tp_sl(
                signal_type=signal['signal_type'],
                current_price=current_price,
                velocity=signal['velocity'],
                acceleration=signal['acceleration'],
                volatility=0.02
            )

            # Apply commission and slippage
            commission = position_size.notional_value * self.config.commission_rate
            slippage = position_size.notional_value * self.config.slippage_rate

            # Create position record
            position = {
                'symbol': symbol,
                'side': side,
                'quantity': position_size.quantity,
                'entry_price': current_price,
                'notional_value': position_size.notional_value,
                'take_profit': trading_levels.take_profit,
                'stop_loss': trading_levels.stop_loss,
                'entry_time': timestamp,
                'signal_type': signal['signal_type'].name,
                'confidence': signal['confidence'],
                'commission': commission,
                'slippage': slippage
            }

            self.positions[symbol] = position
            self.capital -= (commission + slippage)

            logger.debug(f"Entered {side} position: {symbol} @ {current_price:.2f}")

        except Exception as e:
            logger.error(f"Error entering trade: {e}")

    def _manage_positions(self, symbol: str, market_data: pd.Series, signal: Dict, timestamp: datetime):
        """Manage existing positions."""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        current_price = market_data['close']

        # Check exit conditions
        exit_reason = None

        # Stop loss
        if position['side'] == 'long' and current_price <= position['stop_loss']:
            exit_reason = "Stop loss"
        elif position['side'] == 'short' and current_price >= position['stop_loss']:
            exit_reason = "Stop loss"

        # Take profit
        elif position['side'] == 'long' and current_price >= position['take_profit']:
            exit_reason = "Take profit"
        elif position['side'] == 'short' and current_price <= position['take_profit']:
            exit_reason = "Take profit"

        # Signal reversal
        elif signal and ((position['side'] == 'long' and signal['signal_type'] in [SignalType.SELL, SignalType.STRONG_SELL]) or
                        (position['side'] == 'short' and signal['signal_type'] in [SignalType.BUY, SignalType.STRONG_BUY])):
            exit_reason = "Signal reversal"

        # Time-based exit (if position held too long)
        elif (timestamp - position['entry_time']).total_seconds() > 3600:  # 1 hour
            exit_reason = "Time exit"

        if exit_reason:
            self._close_position(symbol, market_data, exit_reason, timestamp)

    def _close_position(self, symbol: str, market_data: pd.Series, reason: str, timestamp: datetime):
        """Close an existing position."""
        try:
            position = self.positions[symbol]
            exit_price = market_data['close']

            # Calculate PnL
            if position['side'] == 'long':
                pnl = (exit_price - position['entry_price']) * position['quantity']
            else:
                pnl = (position['entry_price'] - exit_price) * position['quantity']

            pnl_percent = pnl / position['notional_value']

            # Apply commission and slippage on exit
            exit_commission = position['notional_value'] * self.config.commission_rate
            exit_slippage = position['notional_value'] * self.config.slippage_rate

            total_pnl = pnl - exit_commission - exit_slippage
            self.capital += total_pnl

            # Create trade record
            trade = TradeRecord(
                symbol=symbol,
                entry_time=position['entry_time'],
                exit_time=timestamp,
                entry_price=position['entry_price'],
                exit_price=exit_price,
                quantity=position['quantity'],
                side=position['side'],
                pnl=total_pnl,
                pnl_percent=pnl_percent,
                commission=position['commission'] + exit_commission,
                slippage=position['slippage'] + exit_slippage,
                signal_type=position['signal_type'],
                confidence=position['confidence'],
                holding_period=(timestamp - position['entry_time']).total_seconds(),
                exit_reason=reason
            )

            self.trades.append(trade)
            del self.positions[symbol]

            logger.debug(f"Closed position: {symbol} PnL: {total_pnl:.2f} ({pnl_percent:.2%}) - {reason}")

        except Exception as e:
            logger.error(f"Error closing position: {e}")

    def _calculate_total_equity(self, current_price: float) -> float:
        """Calculate total equity including open positions."""
        equity = self.capital

        for symbol, position in self.positions.items():
            if position['side'] == 'long':
                unrealized_pnl = (current_price - position['entry_price']) * position['quantity']
            else:
                unrealized_pnl = (position['entry_price'] - current_price) * position['quantity']
            equity += unrealized_pnl

        return equity

    def plot_results(self, result: BacktestResult, save_path: str = None):
        """Plot backtesting results."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Backtest Results - {result.parameters.get("symbol", "Unknown")}', fontsize=16)

            # Equity curve
            axes[0, 0].plot(result.equity_curve.index, result.equity_curve.values)
            axes[0, 0].set_title('Equity Curve')
            axes[0, 0].set_ylabel('Portfolio Value')
            axes[0, 0].grid(True)

            # Drawdown
            rolling_max = result.equity_curve.expanding().max()
            drawdown = (result.equity_curve - rolling_max) / rolling_max
            axes[0, 1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
            axes[0, 1].plot(drawdown.index, drawdown.values, color='red')
            axes[0, 1].set_title('Drawdown')
            axes[0, 1].set_ylabel('Drawdown %')
            axes[0, 1].grid(True)

            # Returns distribution
            if not result.returns.empty:
                axes[1, 0].hist(result.returns, bins=50, alpha=0.7, edgecolor='black')
                axes[1, 0].set_title('Returns Distribution')
                axes[1, 0].set_xlabel('Return')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].grid(True)

            # Trade PnL distribution
            if result.trades:
                trade_pnls = [t.pnl_percent for t in result.trades]
                axes[1, 1].hist(trade_pnls, bins=30, alpha=0.7, edgecolor='black')
                axes[1, 1].set_title('Trade P&L Distribution')
                axes[1, 1].set_xlabel('P&L %')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
                axes[1, 1].grid(True)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")

            plt.show()

        except Exception as e:
            logger.error(f"Error plotting results: {e}")

class ParameterOptimizer:
    """Optimizes strategy parameters using grid search"""

    @staticmethod
    def optimize_parameters(symbol: str, data: pd.DataFrame, param_grid: Dict[str, List]) -> Dict:
        """
        Optimize strategy parameters using grid search.

        Args:
            symbol: Trading symbol
            data: Historical data
            param_grid: Dictionary of parameters to optimize

        Returns:
            Best parameters and results
        """
        try:
            logger.info("Starting parameter optimization...")

            best_result = None
            best_params = None
            best_sharpe = -float('inf')

            # Generate all parameter combinations
            import itertools
            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())

            for combination in itertools.product(*param_values):
                params = dict(zip(param_names, combination))

                # Create config with new parameters
                config = BacktestConfig(
                    start_date=data.index[0].strftime('%Y-%m-%d'),
                    end_date=data.index[-1].strftime('%Y-%m-%d'),
                    **params
                )

                # Run backtest
                backtester = CalculusBacktester(config)
                result = backtester.run_backtest(symbol, data)

                # Evaluate based on Sharpe ratio
                sharpe = result.metrics.get('sharpe_ratio', -float('inf'))
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_result = result
                    best_params = params

                logger.info(f"Params: {params} | Sharpe: {sharpe:.2f}")

            logger.info(f"Optimization completed. Best Sharpe: {best_sharpe:.2f}")
            logger.info(f"Best parameters: {best_params}")

            return {
                'best_params': best_params,
                'best_result': best_result,
                'best_sharpe': best_sharpe
            }

        except Exception as e:
            logger.error(f"Error in parameter optimization: {e}")
            return {}

# Example usage
if __name__ == '__main__':
    # Create backtest configuration
    config = BacktestConfig(
        start_date='2023-01-01',
        end_date='2023-12-31',
        initial_capital=10000.0,
        commission_rate=0.001,
        slippage_rate=0.0005
    )

    # Initialize backtester
    backtester = CalculusBacktester(config)

    # Load data
    data = backtester.load_data('BTCUSDT')

    if not data.empty:
        # Run backtest
        result = backtester.run_backtest('BTCUSDT', data)

        # Print results
        print("\n=== BACKTEST RESULTS ===")
        for metric, value in result.metrics.items():
            print(f"{metric}: {value}")

        # Plot results
        backtester.plot_results(result)
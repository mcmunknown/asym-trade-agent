#!/usr/bin/env python3
"""
Comprehensive Backtesting and Validation System for Quantitative Models

This system provides institutional-grade backtesting capabilities for:
1. Position sizing models validation
2. Economic model performance testing
3. Strategy optimization and parameter tuning
4. Monte Carlo simulation for robustness testing
5. Performance attribution analysis
6. Risk-adjusted metrics calculation
7. Scenario analysis and stress testing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters"""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    commission_rate: float = 0.001  # 0.1%
    slippage_rate: float = 0.0005   # 0.05%
    max_position_size: float = 0.3   # 30% of portfolio
    min_position_size: float = 0.01  # 1% of portfolio
    rebalance_frequency: str = 'daily'  # daily, weekly, monthly
    benchmark: str = 'BTCUSDT'  # Benchmark asset

@dataclass
class TradeResult:
    """Result of a single trade"""
    symbol: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    position_size: float
    leverage: float
    pnl: float
    pnl_pct: float
    holding_period: int  # days
    trade_type: str  # LONG, SHORT
    exit_reason: str  # TP, SL, TIME, SIGNAL

@dataclass
class BacktestResults:
    """Comprehensive backtest results"""
    trades: List[TradeResult]
    equity_curve: pd.Series
    returns: pd.Series
    benchmark_returns: pd.Series

    # Performance metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_holding_period: float

    # Risk metrics
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional VaR 95%
    beta: float
    alpha: float
    information_ratio: float

class PositionSizingBacktester:
    """
    Backtester specifically for position sizing models
    """

    def __init__(self, config: BacktestConfig):
        """
        Initialize position sizing backtester

        Args:
            config: Backtest configuration
        """
        self.config = config
        self.reset_results()

    def reset_results(self):
        """Reset backtest results"""
        self.trades = []
        self.equity_curve = []
        self.current_capital = self.config.initial_capital
        self.current_positions = {}

    def backtest_position_sizing_strategy(
        self,
        price_data: Dict[str, pd.DataFrame],
        signals: Dict[str, pd.DataFrame],
        position_sizing_model: Any,
        economic_model: Any = None
    ) -> BacktestResults:
        """
        Backtest a position sizing strategy

        Args:
            price_data: Historical price data for all assets
            signals: Trading signals for all assets
            position_sizing_model: Position sizing model to test
            economic_model: Economic model for market context

        Returns:
            Comprehensive backtest results
        """
        logger.info(f"Starting backtest from {self.config.start_date} to {self.config.end_date}")

        # Prepare data
        all_dates = pd.date_range(
            start=self.config.start_date,
            end=self.config.end_date,
            freq='D'
        )

        equity_values = [self.config.initial_capital]
        capital_history = [self.config.initial_capital]

        # Initialize portfolio
        portfolio = {
            'cash': self.config.initial_capital,
            'positions': {},
            'total_value': self.config.initial_capital
        }

        for date in all_dates:
            try:
                # Get data for current date
                daily_data = self._get_daily_data(date, price_data, signals)

                if not daily_data:
                    equity_values.append(portfolio['total_value'])
                    continue

                # Update portfolio values
                portfolio = self._update_portfolio_values(portfolio, daily_data)

                # Generate trading decisions
                trades_to_execute = self._generate_trades(
                    date, portfolio, daily_data, position_sizing_model, economic_model
                )

                # Execute trades
                portfolio = self._execute_trades(portfolio, trades_to_execute, daily_data)

                # Record equity
                equity_values.append(portfolio['total_value'])
                capital_history.append(portfolio['cash'])

                # Check for exits (TP/SL/time)
                portfolio, completed_trades = self._check_exit_conditions(
                    portfolio, daily_data, date
                )
                self.trades.extend(completed_trades)

            except Exception as e:
                logger.error(f"Error in backtest on {date}: {str(e)}")
                equity_values.append(portfolio['total_value'])
                continue

        # Calculate results
        results = self._calculate_backtest_results(
            equity_values, price_data[self.config.benchmark]
        )

        logger.info(f"Backtest completed: Total Return: {results.total_return:.2%}, Sharpe: {results.sharpe_ratio:.2f}")

        return results

    def _get_daily_data(
        self,
        date: datetime,
        price_data: Dict[str, pd.DataFrame],
        signals: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict]:
        """Get daily market data and signals"""
        daily_data = {}

        for symbol in price_data.keys():
            try:
                if date in price_data[symbol].index:
                    price_info = price_data[symbol].loc[date]
                    signal_info = signals.get(symbol, pd.DataFrame())

                    if date in signal_info.index:
                        signal_data = signal_info.loc[date].to_dict()
                    else:
                        signal_data = {}

                    daily_data[symbol] = {
                        'price': price_info.get('close', 0),
                        'volume': price_info.get('volume', 0),
                        'high': price_info.get('high', 0),
                        'low': price_info.get('low', 0),
                        'volatility': price_info.get('volatility', 0),
                        'signals': signal_data
                    }
            except Exception as e:
                logger.warning(f"Error getting data for {symbol} on {date}: {str(e)}")
                continue

        return daily_data

    def _update_portfolio_values(
        self,
        portfolio: Dict,
        daily_data: Dict[str, Dict]
    ) -> Dict:
        """Update portfolio values based on current prices"""
        positions_value = 0

        for symbol, position in portfolio['positions'].items():
            if symbol in daily_data:
                current_price = daily_data[symbol]['price']
                position_value = position['quantity'] * current_price
                positions_value += position_value

                # Update position value
                position['current_value'] = position_value
                position['current_price'] = current_price
                position['unrealized_pnl'] = position_value - position['cost_basis']

        portfolio['total_value'] = portfolio['cash'] + positions_value
        return portfolio

    def _generate_trades(
        self,
        date: datetime,
        portfolio: Dict,
        daily_data: Dict[str, Dict],
        position_sizing_model: Any,
        economic_model: Any
    ) -> List[Dict]:
        """Generate trading signals and position sizes"""
        trades = []

        for symbol, data in daily_data.items():
            try:
                # Check if we already have a position
                has_position = symbol in portfolio['positions']

                # Get signal
                signal = data['signals'].get('signal', 'HOLD')
                signal_strength = data['signals'].get('strength', 0)

                # Skip if signal is weak
                if abs(signal_strength) < 0.3:
                    continue

                # Generate trade signal based on position and signal
                if not has_position and signal == 'BUY' and signal_strength > 0.5:
                    # Create market data for position sizing
                    market_data = self._create_market_data_for_sizing(
                        symbol, data, portfolio, economic_model
                    )

                    # Calculate position size
                    position_size, sizing_details = position_sizing_model.calculate_optimal_position_size(
                        symbol,
                        market_data['asset_metrics'],
                        market_data['portfolio_metrics'],
                        market_data['market_psychology'],
                        market_data['historical_data']
                    )

                    # Apply constraints
                    max_position = portfolio['total_value'] * self.config.max_position_size
                    min_position = portfolio['total_value'] * self.config.min_position_size

                    position_size = max(min_position, min(position_size, max_position))

                    # Calculate quantity
                    price = data['price']
                    quantity = position_size / price

                    # Apply slippage
                    execution_price = price * (1 + self.config.slippage_rate)

                    trades.append({
                        'symbol': symbol,
                        'action': 'BUY',
                        'quantity': quantity,
                        'price': execution_price,
                        'position_size': position_size,
                        'leverage': sizing_details.get('recommended_leverage', 1.0),
                        'date': date,
                        'sizing_details': sizing_details
                    })

                elif has_position and signal == 'SELL':
                    # Exit signal
                    position = portfolio['positions'][symbol]
                    trades.append({
                        'symbol': symbol,
                        'action': 'SELL',
                        'quantity': position['quantity'],
                        'price': data['price'],
                        'date': date,
                        'exit_reason': 'SIGNAL'
                    })

            except Exception as e:
                logger.error(f"Error generating trade for {symbol}: {str(e)}")
                continue

        return trades

    def _create_market_data_for_sizing(
        self,
        symbol: str,
        daily_data: Dict,
        portfolio: Dict,
        economic_model: Any
    ) -> Dict:
        """Create market data structure for position sizing model"""
        # This is a simplified version - in real implementation, would use comprehensive data

        from quantitative_position_sizing import AssetMetrics, PortfolioMetrics, MarketPsychology

        asset_metrics = AssetMetrics(
            symbol=symbol,
            current_price=daily_data['price'],
            volatility=daily_data['volatility'],
            atr_14d=daily_data['price'] * 0.05,  # Estimate
            beta=1.0,  # Would calculate from historical data
            sharpe_30d=1.0,  # Would calculate
            max_dd_30d=-0.1,  # Would calculate
            correlation_btc=0.8,  # Would calculate
            liquidity_score=70,  # Would calculate
            funding_rate=0.0001,  # Would fetch
            open_interest_change=0.05  # Would fetch
        )

        positions = {k: v['cost_basis'] for k, v in portfolio['positions'].items()}
        portfolio_metrics = PortfolioMetrics(
            total_value=portfolio['total_value'],
            available_cash=portfolio['cash'],
            positions=positions,
            portfolio_volatility=0.6,  # Would calculate
            portfolio_beta=1.0,  # Would calculate
            max_drawdown=-0.05,  # Would calculate
            sharpe_ratio=1.0,  # Would calculate
            correlation_matrix=pd.DataFrame()  # Would create
        )

        market_psychology = MarketPsychology(
            fear_greed_index=50,  # Would calculate
            fomo_intensity=30,  # Would calculate
            sentiment_score=0,  # Would calculate
            social_volume=1.0,  # Would calculate
            volatility_premium=0.1  # Would calculate
        )

        historical_data = {
            'price': pd.Series([daily_data['price']] * 30),  # Would use real historical data
            'volume': pd.Series([daily_data['volume']] * 30),
            'volatility': pd.Series([daily_data['volatility']] * 30)
        }

        return {
            'asset_metrics': asset_metrics,
            'portfolio_metrics': portfolio_metrics,
            'market_psychology': market_psychology,
            'historical_data': historical_data
        }

    def _execute_trades(
        self,
        portfolio: Dict,
        trades: List[Dict],
        daily_data: Dict[str, Dict]
    ) -> Dict:
        """Execute trades and update portfolio"""
        for trade in trades:
            try:
                symbol = trade['symbol']
                action = trade['action']
                quantity = trade['quantity']
                price = trade['price']

                trade_value = quantity * price
                commission = trade_value * self.config.commission_rate

                if action == 'BUY':
                    # Check if enough cash
                    total_cost = trade_value + commission
                    if total_cost <= portfolio['cash']:
                        # Execute buy
                        portfolio['cash'] -= total_cost

                        if symbol not in portfolio['positions']:
                            portfolio['positions'][symbol] = {
                                'quantity': 0,
                                'cost_basis': 0,
                                'entry_date': trade['date'],
                                'entry_price': price,
                                'leverage': trade.get('leverage', 1.0),
                                'stop_loss': price * 0.95,  # 5% stop loss
                                'take_profit': price * 1.10,  # 10% take profit
                                'sizing_details': trade.get('sizing_details', {})
                            }

                        # Update position
                        position = portfolio['positions'][symbol]
                        old_quantity = position['quantity']
                        old_cost_basis = position['cost_basis']

                        position['quantity'] += quantity
                        position['cost_basis'] += trade_value

                        # Average down/up
                        if position['quantity'] > 0:
                            position['avg_price'] = position['cost_basis'] / position['quantity']

                elif action == 'SELL':
                    if symbol in portfolio['positions']:
                        position = portfolio['positions'][symbol]

                        # Can only sell what we have
                        sell_quantity = min(quantity, position['quantity'])
                        sell_value = sell_quantity * price
                        commission = sell_value * self.config.commission_rate

                        # Execute sell
                        portfolio['cash'] += sell_value - commission

                        # Update position
                        position['quantity'] -= sell_quantity
                        position['cost_basis'] -= (sell_quantity / (sell_quantity + position['quantity'])) * position['cost_basis'] if position['quantity'] > 0 else position['cost_basis']

                        # Remove position if fully closed
                        if position['quantity'] <= 0.0001:  # Allow for floating point errors
                            del portfolio['positions'][symbol]

            except Exception as e:
                logger.error(f"Error executing trade {trade}: {str(e)}")
                continue

        return portfolio

    def _check_exit_conditions(
        self,
        portfolio: Dict,
        daily_data: Dict[str, Dict],
        date: datetime
    ) -> Tuple[Dict, List[TradeResult]]:
        """Check and execute exit conditions (TP/SL/time)"""
        completed_trades = []
        positions_to_close = []

        for symbol, position in portfolio['positions'].items():
            if symbol not in daily_data:
                continue

            current_price = daily_data[symbol]['price']
            exit_reason = None

            # Check stop loss
            if current_price <= position['stop_loss']:
                exit_reason = 'STOP_LOSS'

            # Check take profit
            elif current_price >= position['take_profit']:
                exit_reason = 'TAKE_PROFIT'

            # Check time-based exit (e.g., 3 days)
            elif (date - position['entry_date']).days >= 3:
                exit_reason = 'TIME'

            # If exit condition met, prepare trade
            if exit_reason:
                positions_to_close.append({
                    'symbol': symbol,
                    'quantity': position['quantity'],
                    'price': current_price,
                    'exit_reason': exit_reason,
                    'position': position
                })

        # Execute exits
        for exit_trade in positions_to_close:
            symbol = exit_trade['symbol']
            position = exit_trade['position']
            quantity = exit_trade['quantity']
            price = exit_trade['price']

            # Calculate P&L
            sell_value = quantity * price
            commission = sell_value * self.config.commission_rate
            pnl = sell_value - commission - position['cost_basis']
            pnl_pct = pnl / position['cost_basis']

            # Create trade result
            trade_result = TradeResult(
                symbol=symbol,
                entry_date=position['entry_date'],
                exit_date=date,
                entry_price=position['entry_price'],
                exit_price=price,
                position_size=position['cost_basis'],
                leverage=position.get('leverage', 1.0),
                pnl=pnl,
                pnl_pct=pnl_pct,
                holding_period=(date - position['entry_date']).days,
                trade_type='LONG',  # Assuming long positions for now
                exit_reason=exit_trade['exit_reason']
            )

            completed_trades.append(trade_result)

            # Update portfolio
            portfolio['cash'] += sell_value - commission
            del portfolio['positions'][symbol]

        return portfolio, completed_trades

    def _calculate_backtest_results(
        self,
        equity_values: List[float],
        benchmark_prices: pd.Series
    ) -> BacktestResults:
        """Calculate comprehensive backtest results"""
        equity_series = pd.Series(equity_values)

        # Calculate returns
        returns = equity_series.pct_change().dropna()

        # Calculate benchmark returns
        benchmark_returns = benchmark_prices.pct_change().dropna()
        benchmark_returns = benchmark_returns.reindex(returns.index, method='ffill')

        # Performance metrics
        total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1
        n_days = len(equity_series)
        annualized_return = (1 + total_return) ** (252 / n_days) - 1

        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252)
        sortino_ratio = annualized_return / downside_volatility if downside_volatility > 0 else 0

        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Trade statistics
        if self.trades:
            total_trades = len(self.trades)
            winning_trades = len([t for t in self.trades if t.pnl > 0])
            losing_trades = total_trades - winning_trades
            win_rate = winning_trades / total_trades if total_trades > 0 else 0

            wins = [t.pnl for t in self.trades if t.pnl > 0]
            losses = [t.pnl for t in self.trades if t.pnl < 0]

            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            profit_factor = sum(wins) / abs(sum(losses)) if losses else float('inf')

            avg_holding_period = np.mean([t.holding_period for t in self.trades]) if self.trades else 0
        else:
            total_trades = winning_trades = losing_trades = 0
            win_rate = avg_win = avg_loss = profit_factor = avg_holding_period = 0

        # Risk metrics
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()

        # Beta and alpha
        if len(returns) == len(benchmark_returns) and len(returns) > 30:
            benchmark_returns_aligned = benchmark_returns.reindex(returns.index).dropna()
            returns_aligned = returns.reindex(benchmark_returns_aligned.index).dropna()

            if len(returns_aligned) > 30:
                covariance = np.cov(returns_aligned, benchmark_returns_aligned)[0, 1]
                benchmark_variance = np.var(benchmark_returns_aligned)
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 0

                # Calculate risk-free rate (assuming 2% annual)
                risk_free_rate = 0.02
                daily_rf = risk_free_rate / 252

                alpha = (returns_aligned.mean() - daily_rf) - beta * (benchmark_returns_aligned.mean() - daily_rf)
                alpha = alpha * 252  # Annualize

                # Information ratio
                tracking_error = (returns_aligned - benchmark_returns_aligned).std() * np.sqrt(252)
                information_ratio = (alpha / 252) / (tracking_error / np.sqrt(252)) if tracking_error > 0 else 0
            else:
                beta = alpha = information_ratio = 0
        else:
            beta = alpha = information_ratio = 0

        return BacktestResults(
            trades=self.trades,
            equity_curve=equity_series,
            returns=returns,
            benchmark_returns=benchmark_returns,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            avg_holding_period=avg_holding_period,
            var_95=var_95,
            cvar_95=cvar_95,
            beta=beta,
            alpha=alpha,
            information_ratio=information_ratio
        )

class MonteCarloValidator:
    """
    Monte Carlo simulation for robustness testing
    """

    def __init__(self, num_simulations: int = 1000):
        """
        Initialize Monte Carlo validator

        Args:
            num_simulations: Number of simulations to run
        """
        self.num_simulations = num_simulations

    def validate_strategy_robustness(
        self,
        backtest_results: BacktestResults,
        scenarios: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, Any]:
        """
        Validate strategy robustness using Monte Carlo simulation

        Args:
            backtest_results: Original backtest results
            scenarios: Custom scenarios to test

        Returns:
            Robustness validation results
        """
        logger.info(f"Starting Monte Carlo validation with {self.num_simulations} simulations")

        if scenarios is None:
            scenarios = self._generate_default_scenarios()

        results = {}

        for scenario_name, scenario_params in scenarios.items():
            logger.info(f"Testing scenario: {scenario_name}")

            scenario_results = self._run_scenario_simulation(
                backtest_results, scenario_params
            )
            results[scenario_name] = scenario_results

        # Calculate robustness metrics
        robustness_metrics = self._calculate_robustness_metrics(results, backtest_results)

        logger.info("Monte Carlo validation completed")

        return {
            'scenario_results': results,
            'robustness_metrics': robustness_metrics,
            'recommendations': self._generate_robustness_recommendations(robustness_metrics)
        }

    def _generate_default_scenarios(self) -> Dict[str, Dict]:
        """Generate default stress test scenarios"""
        return {
            'base_case': {
                'return_multiplier': 1.0,
                'volatility_multiplier': 1.0,
                'slippage_multiplier': 1.0,
                'commission_multiplier': 1.0
            },
            'high_volatility': {
                'return_multiplier': 0.8,
                'volatility_multiplier': 2.0,
                'slippage_multiplier': 1.5,
                'commission_multiplier': 1.0
            },
            'low_volatility': {
                'return_multiplier': 1.2,
                'volatility_multiplier': 0.5,
                'slippage_multiplier': 0.8,
                'commission_multiplier': 1.0
            },
            'high_costs': {
                'return_multiplier': 0.9,
                'volatility_multiplier': 1.0,
                'slippage_multiplier': 2.0,
                'commission_multiplier': 2.0
            },
            'bear_market': {
                'return_multiplier': 0.5,
                'volatility_multiplier': 1.5,
                'slippage_multiplier': 1.2,
                'commission_multiplier': 1.0
            },
            'crisis': {
                'return_multiplier': 0.2,
                'volatility_multiplier': 3.0,
                'slippage_multiplier': 2.0,
                'commission_multiplier': 1.5
            }
        }

    def _run_scenario_simulation(
        self,
        backtest_results: BacktestResults,
        scenario_params: Dict
    ) -> Dict[str, float]:
        """Run single scenario simulation"""
        returns = backtest_results.returns
        returns_adj = returns * scenario_params['return_multiplier']

        # Add noise based on volatility multiplier
        noise_factor = (scenario_params['volatility_multiplier'] - 1) * 0.5
        returns_adj += np.random.normal(0, noise_factor * returns.std(), len(returns))

        # Apply cost adjustments
        total_return = (1 + returns_adj).prod() - 1
        cost_impact = (scenario_params['slippage_multiplier'] + scenario_params['commission_multiplier'] - 2) * 0.01
        total_return -= cost_impact

        # Calculate metrics
        volatility_adj = returns_adj.std() * np.sqrt(252)
        sharpe_adj = total_return / volatility_adj if volatility_adj > 0 else 0

        # Calculate drawdown
        cumulative_returns = (1 + returns_adj).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown_adj = drawdown.min()

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_adj,
            'max_drawdown': max_drawdown_adj,
            'volatility': volatility_adj,
            'calmar_ratio': total_return / abs(max_drawdown_adj) if max_drawdown_adj != 0 else 0
        }

    def _calculate_robustness_metrics(
        self,
        scenario_results: Dict[str, Dict],
        base_results: BacktestResults
    ) -> Dict[str, float]:
        """Calculate robustness metrics across scenarios"""

        # Collect all scenario results
        all_returns = [result['total_return'] for result in scenario_results.values()]
        all_sharpes = [result['sharpe_ratio'] for result in scenario_results.values()]
        all_drawdowns = [result['max_drawdown'] for result in scenario_results.values()]

        # Robustness metrics
        return_stability = 1 - (np.std(all_returns) / abs(np.mean(all_returns))) if np.mean(all_returns) != 0 else 0
        sharpe_consistency = min(all_sharpes) / max(all_sharpes) if max(all_sharpes) > 0 else 0
        drawdown_control = 1 - (abs(min(all_drawdowns)) / abs(base_results.max_drawdown)) if base_results.max_drawdown != 0 else 0

        # Overall robustness score
        robustness_score = (return_stability + sharpe_consistency + drawdown_control) / 3

        return {
            'return_stability': return_stability,
            'sharpe_consistency': sharpe_consistency,
            'drawdown_control': drawdown_control,
            'overall_robustness': robustness_score,
            'worst_case_return': min(all_returns),
            'worst_case_sharpe': min(all_sharpes),
            'worst_case_drawdown': min(all_drawdowns)
        }

    def _generate_robustness_recommendations(
        self,
        robustness_metrics: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations based on robustness analysis"""
        recommendations = []

        if robustness_metrics['overall_robustness'] < 0.7:
            recommendations.append("Strategy shows low robustness - consider reducing position sizes")

        if robustness_metrics['return_stability'] < 0.6:
            recommendations.append("Returns are highly variable - implement better risk controls")

        if robustness_metrics['sharpe_consistency'] < 0.5:
            recommendations.append("Risk-adjusted returns inconsistent - review strategy logic")

        if robustness_metrics['drawdown_control'] < 0.6:
            recommendations.append("Drawdowns vary significantly - strengthen risk management")

        if robustness_metrics['worst_case_return'] < -0.2:
            recommendations.append("Potential for large losses - implement stricter stop losses")

        if robustness_metrics['worst_case_sharpe'] < 0.5:
            recommendations.append("Risk-adjusted performance may degrade - review position sizing")

        if not recommendations:
            recommendations.append("Strategy shows good robustness across scenarios")

        return recommendations

class PerformanceAnalyzer:
    """
    Performance analysis and attribution for quantitative strategies
    """

    def __init__(self):
        """Initialize performance analyzer"""
        pass

    def analyze_performance_attribution(
        self,
        backtest_results: BacktestResults,
        benchmark_results: Optional[BacktestResults] = None
    ) -> Dict[str, Any]:
        """
        Analyze performance attribution

        Args:
            backtest_results: Strategy backtest results
            benchmark_results: Benchmark backtest results (optional)

        Returns:
            Performance attribution analysis
        """
        attribution = {}

        # Trade analysis
        trade_analysis = self._analyze_trades(backtest_results.trades)
        attribution['trade_analysis'] = trade_analysis

        # Monthly performance
        monthly_performance = self._calculate_monthly_performance(backtest_results)
        attribution['monthly_performance'] = monthly_performance

        # Rolling metrics
        rolling_metrics = self._calculate_rolling_metrics(backtest_results)
        attribution['rolling_metrics'] = rolling_metrics

        # Benchmark comparison
        if benchmark_results:
            benchmark_comparison = self._compare_to_benchmark(
                backtest_results, benchmark_results
            )
            attribution['benchmark_comparison'] = benchmark_comparison

        # Risk analysis
        risk_analysis = self._analyze_risk_characteristics(backtest_results)
        attribution['risk_analysis'] = risk_analysis

        return attribution

    def _analyze_trades(self, trades: List[TradeResult]) -> Dict[str, Any]:
        """Analyze individual trade characteristics"""
        if not trades:
            return {'message': 'No trades to analyze'}

        # Trade duration analysis
        holding_periods = [t.holding_period for t in trades]

        # P&L distribution
        pnls = [t.pnl for t in trades]
        pnl_pcts = [t.pnl_pct for t in trades]

        # Exit reason analysis
        exit_reasons = {}
        for trade in trades:
            reason = trade.exit_reason
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

        # Position size analysis
        position_sizes = [t.position_size for t in trades]

        return {
            'total_trades': len(trades),
            'avg_holding_period': np.mean(holding_periods),
            'median_holding_period': np.median(holding_periods),
            'total_pnl': sum(pnls),
            'avg_pnl': np.mean(pnls),
            'median_pnl': np.median(pnls),
            'avg_pnl_pct': np.mean(pnl_pcts),
            'median_pnl_pct': np.median(pnl_pcts),
            'exit_reasons': exit_reasons,
            'avg_position_size': np.mean(position_sizes),
            'position_size_std': np.std(position_sizes)
        }

    def _calculate_monthly_performance(
        self,
        backtest_results: BacktestResults
    ) -> pd.DataFrame:
        """Calculate monthly performance breakdown"""
        equity_curve = backtest_results.equity_curve

        # Resample to monthly
        monthly_equity = equity_curve.resample('M').last()
        monthly_returns = monthly_equity.pct_change().dropna()

        # Create monthly performance dataframe
        performance_data = []
        for date, ret in monthly_returns.items():
            performance_data.append({
                'month': date.strftime('%Y-%m'),
                'return': ret,
                'cumulative_return': (monthly_equity.loc[date] / monthly_equity.iloc[0]) - 1
            })

        return pd.DataFrame(performance_data)

    def _calculate_rolling_metrics(
        self,
        backtest_results: BacktestResults,
        window: int = 30
    ) -> Dict[str, pd.Series]:
        """Calculate rolling performance metrics"""
        returns = backtest_results.returns

        rolling_metrics = {
            'rolling_sharpe': returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252),
            'rolling_volatility': returns.rolling(window).std() * np.sqrt(252),
            'rolling_drawdown': self._calculate_rolling_drawdown(returns),
            'rolling_win_rate': self._calculate_rolling_win_rate(returns, window)
        }

        return rolling_metrics

    def _calculate_rolling_drawdown(self, returns: pd.Series) -> pd.Series:
        """Calculate rolling drawdown"""
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown

    def _calculate_rolling_win_rate(
        self,
        returns: pd.Series,
        window: int
    ) -> pd.Series:
        """Calculate rolling win rate"""
        wins = (returns > 0).rolling(window).mean()
        return wins

    def _compare_to_benchmark(
        self,
        strategy_results: BacktestResults,
        benchmark_results: BacktestResults
    ) -> Dict[str, Any]:
        """Compare strategy performance to benchmark"""

        # Performance comparison
        comparison = {
            'excess_return': strategy_results.total_return - benchmark_results.total_return,
            'excess_sharpe': strategy_results.sharpe_ratio - benchmark_results.sharpe_ratio,
            'excess_volatility': strategy_results.volatility - benchmark_results.volatility,
            'alpha': strategy_results.alpha,
            'beta': strategy_results.beta,
            'information_ratio': strategy_results.information_ratio,
            'tracking_error': (strategy_results.returns - benchmark_results.returns).std() * np.sqrt(252)
        }

        # Drawdown comparison
        comparison['drawdown_improvement'] = (
            strategy_results.max_drawdown - benchmark_results.max_drawdown
        )

        return comparison

    def _analyze_risk_characteristics(
        self,
        backtest_results: BacktestResults
    ) -> Dict[str, Any]:
        """Analyze risk characteristics"""
        returns = backtest_results.returns

        risk_analysis = {
            'value_at_risk_95': backtest_results.var_95,
            'conditional_var_95': backtest_results.cvar_95,
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'tail_ratio': self._calculate_tail_ratio(returns),
            'gain_loss_ratio': self._calculate_gain_loss_ratio(returns),
            'pain_index': self._calculate_pain_index(returns)
        }

        return risk_analysis

    def _calculate_tail_ratio(self, returns: pd.Series) -> float:
        """Calculate tail ratio (95th percentile / 5th percentile)"""
        upper_tail = returns.quantile(0.95)
        lower_tail = returns.quantile(0.05)
        return abs(upper_tail / lower_tail) if lower_tail != 0 else float('inf')

    def _calculate_gain_loss_ratio(self, returns: pd.Series) -> float:
        """Calculate gain to loss ratio"""
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        return gains / losses if losses > 0 else float('inf')

    def _calculate_pain_index(self, returns: pd.Series) -> float:
        """Calculate pain index (average drawdown)"""
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0

if __name__ == "__main__":
    # Example usage
    config = BacktestConfig(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2024, 1, 1),
        initial_capital=10000
    )

    backtester = PositionSizingBacktester(config)

    # Create sample data (in real usage, would load from files/API)
    dates = pd.date_range(start=config.start_date, end=config.end_date, freq='D')
    price_data = {
        'BTCUSDT': pd.DataFrame({
            'close': 50000 + np.random.randn(len(dates)).cumsum() * 500,
            'volume': np.random.rand(len(dates)) * 1000,
            'high': 0,  # Would calculate
            'low': 0,   # Would calculate
            'volatility': 0.02
        }, index=dates)
    }

    signals = {
        'BTCUSDT': pd.DataFrame({
            'signal': np.random.choice(['BUY', 'SELL', 'HOLD'], len(dates)),
            'strength': np.random.rand(len(dates))
        }, index=dates)
    }

    # Run backtest
    # results = backtester.backtest_position_sizing_strategy(
    #     price_data, signals, position_sizing_model
    # )

    print("Backtesting system initialized")
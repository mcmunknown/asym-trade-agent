"""
Enhanced Performance Dashboard
===============================

Real-time performance metrics with Sharpe ratio, AR(1) regime analysis,
and market maker statistics.
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class DashboardMetrics:
    """Complete metrics for performance dashboard"""
    # Trading Performance
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pnl_percent: float = 0.0
    
    # Risk Metrics
    sharpe_ratio: float = 0.0
    current_leverage: float = 1.0
    leverage_mode: str = "Bootstrap"
    max_drawdown: float = 0.0
    
    # AR(1) Regime Analysis
    ar_mean_weight: float = 0.0
    ar_mean_r2: float = 0.0
    ar_strategy_counts: Dict[str, int] = None
    current_regime: str = "NEUTRAL"
    
    # Market Maker Stats (if applicable)
    mm_active: bool = False
    mm_spread_avg: float = 0.0
    mm_edge_captured: float = 0.0
    mm_inventory: float = 0.0
    
    # Error Analysis
    total_errors: int = 0
    top_errors: List[tuple] = None
    
    # System Health
    uptime_seconds: float = 0.0
    signals_generated: int = 0
    execution_rate: float = 0.0


class PerformanceDashboard:
    """
    Enhanced performance dashboard with real-time metrics.
    """
    
    def __init__(self):
        self.start_time = time.time()
        self.last_print_time = 0
        self.print_interval = 60  # Print every 60 seconds
    
    def should_print(self) -> bool:
        """Check if enough time has passed for dashboard update."""
        current_time = time.time()
        if current_time - self.last_print_time >= self.print_interval:
            self.last_print_time = current_time
            return True
        return False
    
    def format_dashboard(self, metrics: DashboardMetrics) -> str:
        """
        Format comprehensive dashboard display.
        
        Args:
            metrics: Dashboard metrics
            
        Returns:
            Formatted dashboard string
        """
        lines = []
        lines.append("\n" + "="*80)
        lines.append("📊 ENHANCED PERFORMANCE DASHBOARD")
        lines.append("="*80)
        
        # Trading Performance
        lines.append("\n🎯 TRADING PERFORMANCE:")
        lines.append(f"   Total Trades: {metrics.total_trades} "
                    f"(W: {metrics.winning_trades}, L: {metrics.losing_trades})")
        lines.append(f"   Win Rate: {metrics.win_rate:.1%} | "
                    f"PnL: ${metrics.total_pnl:.2f} ({metrics.total_pnl_percent:+.1f}%)")
        lines.append(f"   Execution Rate: {metrics.execution_rate:.1%} "
                    f"({metrics.total_trades}/{metrics.signals_generated} signals)")
        
        # Risk & Leverage
        lines.append("\n⚡ RISK & LEVERAGE:")
        lines.append(f"   Sharpe Ratio: {metrics.sharpe_ratio:.2f} | "
                    f"Max Drawdown: {metrics.max_drawdown:.1%}")
        lines.append(f"   Current Leverage: {metrics.current_leverage:.1f}x | "
                    f"Mode: {metrics.leverage_mode}")
        
        if metrics.leverage_mode == "Bootstrap":
            if metrics.total_trades <= 20:
                phase = "Phase 1/3 (Baseline)"
            elif metrics.total_trades <= 50:
                phase = "Phase 2/3 (Ramp-up)"
            elif metrics.total_trades <= 100:
                phase = "Phase 3/3 (Pre-dynamic)"
            else:
                phase = "Ready for Dynamic"
            lines.append(f"   Bootstrap: {phase} - Trade #{metrics.total_trades}/100")
        
        # AR(1) Regime Intelligence
        lines.append("\n🔬 AR(1) REGIME ANALYSIS:")
        lines.append(f"   Current Regime: {metrics.current_regime}")
        lines.append(f"   Mean Weight: {metrics.ar_mean_weight:+.3f} | "
                    f"Mean R²: {metrics.ar_mean_r2:.3f}")
        
        if metrics.ar_strategy_counts:
            strategy_str = ", ".join([f"{k}: {v}" for k, v in metrics.ar_strategy_counts.items()])
            lines.append(f"   Strategies: {strategy_str}")
        
        # Market Maker Stats (if active)
        if metrics.mm_active:
            lines.append("\n💱 MARKET MAKER:")
            lines.append(f"   Avg Spread: {metrics.mm_spread_avg:.2f} bps | "
                        f"Edge Captured: {metrics.mm_edge_captured:.2%}")
            lines.append(f"   Inventory: {metrics.mm_inventory:+.2f} "
                        f"({'⚠️ FLATTEN' if abs(metrics.mm_inventory) > 0.8 else '✓ OK'})")
        
        # Error Analysis
        if metrics.total_errors > 0:
            lines.append("\n⚠️  ERROR ANALYSIS:")
            lines.append(f"   Total Errors: {metrics.total_errors}")
            if metrics.top_errors:
                lines.append("   Top Issues:")
                for error_type, count, pct in metrics.top_errors[:3]:
                    lines.append(f"      • {error_type}: {count} ({pct:.1f}%)")
        
        # System Health
        uptime_mins = metrics.uptime_seconds / 60
        uptime_hours = uptime_mins / 60
        if uptime_hours >= 1:
            uptime_str = f"{uptime_hours:.1f}h"
        else:
            uptime_str = f"{uptime_mins:.0f}m"
        
        lines.append(f"\n⏱️  System Uptime: {uptime_str} | Signals: {metrics.signals_generated}")
        lines.append("="*80 + "\n")
        
        return "\n".join(lines)
    
    def print_dashboard(self, metrics: DashboardMetrics):
        """Print dashboard if interval has passed."""
        if self.should_print():
            print(self.format_dashboard(metrics))
    
    def format_trade_summary(self, symbol: str, side: str, pnl: float, 
                            pnl_pct: float, sharpe: float) -> str:
        """
        Format individual trade summary with Sharpe context.
        
        Args:
            symbol: Trading symbol
            side: Trade side (Buy/Sell)
            pnl: Profit/loss in dollars
            pnl_pct: Profit/loss as percentage
            sharpe: Current Sharpe ratio
            
        Returns:
            Formatted trade summary
        """
        pnl_icon = "✅" if pnl > 0 else "❌"
        sharpe_icon = "🔥" if sharpe > 1.5 else ("⚡" if sharpe > 1.0 else "📊")
        
        lines = []
        lines.append(f"\n{pnl_icon} TRADE CLOSED: {symbol} {side}")
        lines.append(f"   PnL: ${pnl:+.2f} ({pnl_pct:+.1f}%)")
        lines.append(f"   {sharpe_icon} Sharpe: {sharpe:.2f}")
        
        return "\n".join(lines)
    
    def format_signal_banner(self, symbol: str, signal_type: str, 
                            confidence: float, ar_data: Optional[Dict] = None) -> str:
        """
        Format signal banner with AR(1) intelligence.
        
        Args:
            symbol: Trading symbol
            signal_type: Signal type name
            confidence: Signal confidence
            ar_data: AR(1) data (optional)
            
        Returns:
            Formatted signal banner
        """
        lines = []
        lines.append("\n" + "="*70)
        lines.append(f"🎯 SIGNAL: {symbol} | {signal_type} @ {confidence:.1%}")
        
        if ar_data:
            strategy_names = {
                0: "No Trade",
                1: "Mean Reversion ⚖️",
                2: "Momentum Long 📈",
                3: "Momentum Short 📉"
            }
            strategy = strategy_names.get(ar_data.get('ar_strategy', 0), "Unknown")
            lines.append(f"   🔬 AR(1): {strategy} | "
                        f"w={ar_data.get('ar_weight', 0):+.3f} | "
                        f"R²={ar_data.get('ar_r_squared', 0):.3f}")
        
        lines.append("="*70)
        
        return "\n".join(lines)


def calculate_dashboard_metrics(trading_states: Dict, 
                               performance: any,
                               risk_manager: any,
                               start_time: float) -> DashboardMetrics:
    """
    Calculate comprehensive dashboard metrics from trading system state.
    
    Args:
        trading_states: Dictionary of trading states by symbol
        performance: Performance metrics object
        risk_manager: Risk manager object
        start_time: System start time
        
    Returns:
        DashboardMetrics with all calculated values
    """
    metrics = DashboardMetrics()
    
    # Trading performance
    metrics.total_trades = performance.total_trades
    metrics.winning_trades = performance.winning_trades
    metrics.losing_trades = performance.losing_trades
    metrics.win_rate = performance.success_rate
    metrics.total_pnl = performance.total_pnl
    metrics.total_pnl_percent = performance.total_return
    
    # Signals and execution rate
    metrics.signals_generated = sum(state.signal_count for state in trading_states.values())
    metrics.execution_rate = (metrics.total_trades / metrics.signals_generated 
                             if metrics.signals_generated > 0 else 0.0)
    
    # Risk metrics from Sharpe tracker
    if hasattr(risk_manager, 'sharpe_tracker') and risk_manager.sharpe_tracker:
        try:
            metrics.sharpe_ratio = risk_manager.sharpe_tracker.calculate_sharpe()
        except:
            metrics.sharpe_ratio = 0.0
    
    # Leverage info
    total_trades_count = len(risk_manager.trade_history) if hasattr(risk_manager, 'trade_history') else 0
    if total_trades_count <= 100:
        metrics.leverage_mode = "Bootstrap"
        metrics.current_leverage = risk_manager.leverage_bootstrap.get_bootstrap_leverage(total_trades_count)
    else:
        metrics.leverage_mode = "Dynamic (Sharpe-based)"
        if risk_manager.sharpe_tracker.has_sufficient_data():
            metrics.current_leverage = risk_manager.sharpe_tracker.get_recommended_leverage(10.0)
        else:
            metrics.current_leverage = 1.0
    
    # AR(1) regime analysis
    ar_weights = []
    ar_r2s = []
    ar_strategies = defaultdict(int)
    
    for state in trading_states.values():
        if hasattr(state, 'last_signal') and state.last_signal:
            signal = state.last_signal
            if 'ar_weight' in signal:
                ar_weights.append(signal['ar_weight'])
            if 'ar_r_squared' in signal:
                ar_r2s.append(signal['ar_r_squared'])
            if 'ar_strategy' in signal:
                strategy_names = {0: "no_trade", 1: "mean_reversion", 
                                2: "momentum_long", 3: "momentum_short"}
                ar_strategies[strategy_names.get(signal['ar_strategy'], 'unknown')] += 1
    
    if ar_weights:
        metrics.ar_mean_weight = sum(ar_weights) / len(ar_weights)
    if ar_r2s:
        metrics.ar_mean_r2 = sum(ar_r2s) / len(ar_r2s)
    if ar_strategies:
        metrics.ar_strategy_counts = dict(ar_strategies)
    
    # Error analysis
    metrics.total_errors = sum(state.error_count for state in trading_states.values())
    if metrics.total_errors > 0:
        error_summary = defaultdict(int)
        for state in trading_states.values():
            if hasattr(state, 'error_breakdown'):
                for category, count in state.error_breakdown.items():
                    error_summary[category.value] += count
        
        sorted_errors = sorted(error_summary.items(), key=lambda x: x[1], reverse=True)[:5]
        metrics.top_errors = [(cat, count, (count/metrics.total_errors)*100) 
                             for cat, count in sorted_errors]
    
    # System uptime
    metrics.uptime_seconds = time.time() - start_time
    
    return metrics

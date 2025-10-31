"""
INSTITUTIONAL RISK MONITORING DASHBOARD
======================================

Real-time risk monitoring dashboard for institutional trading oversight.
Provides comprehensive visibility into system risk metrics and trading performance.

Features:
- Real-time risk metrics visualization
- Circuit breaker status monitoring
- Trade execution tracking
- Performance analytics
- Alert management
- Emergency controls
"""

import logging
import time
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import threading

from risk_management_system import RiskMetrics, CircuitBreakerState, RiskLevel
from risk_enforcement_layer import RiskEnforcementLayer
from enhanced_trading_engine import EnhancedTradingEngine

logger = logging.getLogger(__name__)

@dataclass
class DashboardMetrics:
    """Dashboard display metrics"""
    timestamp: float
    account_balance: float
    total_exposure_pct: float
    daily_pnl_pct: float
    max_drawdown_pct: float
    open_positions: int
    risk_score: float
    circuit_breaker_state: str
    trades_executed_today: int
    success_rate: float
    total_pnl: float
    active_alerts: List[str]
    system_health: str

class RiskMonitoringDashboard:
    """
    Real-time risk monitoring dashboard
    """

    def __init__(self, risk_enforcement: RiskEnforcementLayer, trading_engine: EnhancedTradingEngine):
        self.risk_enforcement = risk_enforcement
        self.trading_engine = trading_engine

        # Dashboard state
        self.monitoring_active = False
        self.update_interval = 30  # seconds
        self.metrics_history = []
        self.max_history_size = 1000

        # Alert thresholds
        self.alert_thresholds = {
            'risk_score_critical': 80,
            'risk_score_warning': 60,
            'drawdown_critical': 10,
            'drawdown_warning': 5,
            'exposure_critical': 45,
            'exposure_warning': 30,
            'daily_loss_critical': 2,
            'daily_loss_warning': 1
        }

        # Start monitoring thread
        self.monitoring_thread = None

        logger.info("📊 Risk Monitoring Dashboard initialized")

    def start_monitoring(self):
        """Start real-time monitoring"""

        if self.monitoring_active:
            logger.warning("Dashboard monitoring already active")
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        logger.info("📊 Dashboard monitoring started")

    def stop_monitoring(self):
        """Stop real-time monitoring"""

        if not self.monitoring_active:
            return

        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)

        logger.info("📊 Dashboard monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop"""

        logger.info("📊 Entering monitoring loop...")

        while self.monitoring_active:
            try:
                # Collect current metrics
                current_metrics = self._collect_dashboard_metrics()

                # Store in history
                self.metrics_history.append(current_metrics)

                # Trim history
                if len(self.metrics_history) > self.max_history_size:
                    self.metrics_history = self.metrics_history[-self.max_history_size:]

                # Generate alerts if needed
                alerts = self._generate_alerts(current_metrics)

                # Log status update
                self._log_status_update(current_metrics, alerts)

                # Sleep until next update
                time.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"❌ Error in monitoring loop: {str(e)}")
                time.sleep(60)  # Wait longer after errors

    def _collect_dashboard_metrics(self) -> DashboardMetrics:
        """Collect current dashboard metrics"""

        try:
            # Get risk status
            risk_status = self.risk_enforcement.get_risk_status()

            # Get engine status
            engine_status = self.trading_engine.get_engine_status()

            # Get current risk metrics
            current_metrics = self.risk_enforcement.update_risk_metrics(self.trading_engine.bybit_client)

            # Generate alerts
            alerts = self._generate_alerts_internal(current_metrics)

            # Determine system health
            system_health = self._determine_system_health(current_metrics, risk_status)

            return DashboardMetrics(
                timestamp=time.time(),
                account_balance=current_metrics.current_account_balance,
                total_exposure_pct=current_metrics.total_exposure_pct,
                daily_pnl_pct=current_metrics.daily_pnl_pct,
                max_drawdown_pct=current_metrics.max_drawdown_pct,
                open_positions=current_metrics.open_positions_count,
                risk_score=current_metrics.risk_score,
                circuit_breaker_state=risk_status['circuit_breaker_state'],
                trades_executed_today=engine_status['trades_executed'],
                success_rate=engine_status['success_rate'],
                total_pnl=engine_status['total_pnl'],
                active_alerts=alerts,
                system_health=system_health
            )

        except Exception as e:
            logger.error(f"❌ Failed to collect dashboard metrics: {str(e)}")

            # Return safe defaults
            return DashboardMetrics(
                timestamp=time.time(),
                account_balance=0.0,
                total_exposure_pct=0.0,
                daily_pnl_pct=0.0,
                max_drawdown_pct=0.0,
                open_positions=0,
                risk_score=0.0,
                circuit_breaker_state="UNKNOWN",
                trades_executed_today=0,
                success_rate=0.0,
                total_pnl=0.0,
                active_alerts=[f"Error collecting metrics: {str(e)}"],
                system_health="ERROR"
            )

    def _generate_alerts_internal(self, metrics: RiskMetrics) -> List[str]:
        """Generate alerts based on current metrics"""

        alerts = []

        # Risk score alerts
        if metrics.risk_score >= self.alert_thresholds['risk_score_critical']:
            alerts.append(f"🚨 CRITICAL: Risk score {metrics.risk_score:.1f} exceeds threshold")
        elif metrics.risk_score >= self.alert_thresholds['risk_score_warning']:
            alerts.append(f"⚠️ WARNING: Risk score {metrics.risk_score:.1f} elevated")

        # Drawdown alerts
        if abs(metrics.max_drawdown_pct) >= self.alert_thresholds['drawdown_critical']:
            alerts.append(f"🚨 CRITICAL: Drawdown {metrics.max_drawdown_pct:.2f}% exceeds threshold")
        elif abs(metrics.max_drawdown_pct) >= self.alert_thresholds['drawdown_warning']:
            alerts.append(f"⚠️ WARNING: Drawdown {metrics.max_drawdown_pct:.2f}% elevated")

        # Exposure alerts
        if metrics.total_exposure_pct >= self.alert_thresholds['exposure_critical']:
            alerts.append(f"🚨 CRITICAL: Total exposure {metrics.total_exposure_pct:.2f}% exceeds threshold")
        elif metrics.total_exposure_pct >= self.alert_thresholds['exposure_warning']:
            alerts.append(f"⚠️ WARNING: Total exposure {metrics.total_exposure_pct:.2f}% elevated")

        # Daily loss alerts
        if metrics.daily_pnl_pct <= -self.alert_thresholds['daily_loss_critical']:
            alerts.append(f"🚨 CRITICAL: Daily loss {metrics.daily_pnl_pct:.2f}% exceeds threshold")
        elif metrics.daily_pnl_pct <= -self.alert_thresholds['daily_loss_warning']:
            alerts.append(f"⚠️ WARNING: Daily loss {metrics.daily_pnl_pct:.2f}% elevated")

        # Position count alerts
        if metrics.open_positions_count >= 8:
            alerts.append(f"⚠️ WARNING: High position count {metrics.open_positions_count}")

        return alerts

    def _determine_system_health(self, metrics: RiskMetrics, risk_status: Dict) -> str:
        """Determine overall system health status"""

        # Check for critical issues
        if risk_status['circuit_breaker_state'] != 'CLOSED':
            return "CRITICAL"

        if metrics.risk_score >= 80:
            return "CRITICAL"

        if abs(metrics.max_drawdown_pct) >= 10:
            return "CRITICAL"

        if metrics.daily_pnl_pct <= -2:
            return "CRITICAL"

        # Check for warning conditions
        if metrics.risk_score >= 60:
            return "WARNING"

        if abs(metrics.max_drawdown_pct) >= 5:
            return "WARNING"

        if metrics.daily_pnl_pct <= -1:
            return "WARNING"

        if metrics.total_exposure_pct >= 30:
            return "WARNING"

        # System is healthy
        return "HEALTHY"

    def _generate_alerts(self, metrics: DashboardMetrics) -> List[str]:
        """Generate alerts for dashboard display"""

        return metrics.active_alerts

    def _log_status_update(self, metrics: DashboardMetrics, alerts: List[str]):
        """Log periodic status updates"""

        # Log every 5 minutes or when there are alerts
        current_time = time.time()
        log_interval = 300  # 5 minutes

        if alerts or (int(current_time) % log_interval < self.update_interval):
            timestamp_str = datetime.fromtimestamp(metrics.timestamp).strftime('%Y-%m-%d %H:%M:%S')

            logger.info(f"📊 Dashboard Update [{timestamp_str}]:")
            logger.info(f"   System Health: {metrics.system_health}")
            logger.info(f"   Account Balance: ${metrics.account_balance:.2f}")
            logger.info(f"   Total Exposure: {metrics.total_exposure_pct:.2f}%")
            logger.info(f"   Daily PnL: {metrics.daily_pnl_pct:.2f}%")
            logger.info(f"   Max Drawdown: {metrics.max_drawdown_pct:.2f}%")
            logger.info(f"   Risk Score: {metrics.risk_score:.1f}")
            logger.info(f"   Open Positions: {metrics.open_positions}")
            logger.info(f"   Circuit Breaker: {metrics.circuit_breaker_state}")
            logger.info(f"   Trades Today: {metrics.trades_executed_today}")
            logger.info(f"   Success Rate: {metrics.success_rate:.1f}%")

            if alerts:
                logger.warning(f"   Active Alerts ({len(alerts)}):")
                for alert in alerts:
                    logger.warning(f"      - {alert}")

    def get_dashboard_data(self) -> Dict:
        """Get current dashboard data for API/display"""

        if not self.metrics_history:
            return {"status": "No data available"}

        # Get latest metrics
        latest_metrics = self.metrics_history[-1]

        # Prepare chart data
        chart_data = {
            'timestamps': [m.timestamp for m in self.metrics_history[-100:]],  # Last 100 data points
            'account_balance': [m.account_balance for m in self.metrics_history[-100:]],
            'total_exposure_pct': [m.total_exposure_pct for m in self.metrics_history[-100:]],
            'daily_pnl_pct': [m.daily_pnl_pct for m in self.metrics_history[-100:]],
            'risk_score': [m.risk_score for m in self.metrics_history[-100:]],
        }

        # Risk level distribution
        risk_distribution = self._calculate_risk_distribution()

        # Performance summary
        performance_summary = self._calculate_performance_summary()

        return {
            'timestamp': latest_metrics.timestamp,
            'current_metrics': asdict(latest_metrics),
            'chart_data': chart_data,
            'risk_distribution': risk_distribution,
            'performance_summary': performance_summary,
            'alert_thresholds': self.alert_thresholds,
            'monitoring_status': 'ACTIVE' if self.monitoring_active else 'INACTIVE'
        }

    def _calculate_risk_distribution(self) -> Dict:
        """Calculate risk distribution across positions"""

        try:
            engine_status = self.trading_engine.get_engine_status()
            positions = engine_status.get('active_position_details', {})

            risk_levels = {'MINIMAL': 0, 'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'CRITICAL': 0}

            for symbol, position_data in positions.items():
                risk_level = position_data.get('risk_level', 'LOW')
                if risk_level in risk_levels:
                    risk_levels[risk_level] += 1

            return risk_levels

        except Exception as e:
            logger.error(f"❌ Failed to calculate risk distribution: {str(e)}")
            return {'MINIMAL': 0, 'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'CRITICAL': 0}

    def _calculate_performance_summary(self) -> Dict:
        """Calculate performance summary statistics"""

        try:
            if not self.metrics_history:
                return {}

            # Get last 24 hours of data
            cutoff_time = time.time() - 86400  # 24 hours ago
            recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]

            if not recent_metrics:
                return {}

            # Calculate statistics
            balances = [m.account_balance for m in recent_metrics]
            risk_scores = [m.risk_score for m in recent_metrics]
            exposures = [m.total_exposure_pct for m in recent_metrics]

            return {
                'period_hours': 24,
                'starting_balance': balances[0] if balances else 0,
                'ending_balance': balances[-1] if balances else 0,
                'balance_change': (balances[-1] - balances[0]) if len(balances) >= 2 else 0,
                'avg_risk_score': sum(risk_scores) / len(risk_scores) if risk_scores else 0,
                'max_risk_score': max(risk_scores) if risk_scores else 0,
                'avg_exposure_pct': sum(exposures) / len(exposures) if exposures else 0,
                'max_exposure_pct': max(exposures) if exposures else 0,
                'data_points': len(recent_metrics)
            }

        except Exception as e:
            logger.error(f"❌ Failed to calculate performance summary: {str(e)}")
            return {}

    def export_report(self, format: str = 'json') -> str:
        """Export dashboard report"""

        try:
            dashboard_data = self.get_dashboard_data()

            if format.lower() == 'json':
                return json.dumps(dashboard_data, indent=2, default=str)

            elif format.lower() == 'txt':
                # Create text report
                report_lines = []
                report_lines.append("RISK MONITORING DASHBOARD REPORT")
                report_lines.append("=" * 50)
                report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                report_lines.append("")

                # Current status
                metrics = dashboard_data.get('current_metrics', {})
                report_lines.append("CURRENT STATUS")
                report_lines.append("-" * 20)
                report_lines.append(f"System Health: {metrics.get('system_health', 'UNKNOWN')}")
                report_lines.append(f"Account Balance: ${metrics.get('account_balance', 0):.2f}")
                report_lines.append(f"Total Exposure: {metrics.get('total_exposure_pct', 0):.2f}%")
                report_lines.append(f"Daily PnL: {metrics.get('daily_pnl_pct', 0):.2f}%")
                report_lines.append(f"Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
                report_lines.append(f"Risk Score: {metrics.get('risk_score', 0):.1f}")
                report_lines.append(f"Open Positions: {metrics.get('open_positions', 0)}")
                report_lines.append(f"Circuit Breaker: {metrics.get('circuit_breaker_state', 'UNKNOWN')}")
                report_lines.append("")

                # Alerts
                alerts = metrics.get('active_alerts', [])
                if alerts:
                    report_lines.append("ACTIVE ALERTS")
                    report_lines.append("-" * 20)
                    for alert in alerts:
                        report_lines.append(f"  {alert}")
                    report_lines.append("")
                else:
                    report_lines.append("ACTIVE ALERTS: None")
                    report_lines.append("")

                # Performance summary
                perf_summary = dashboard_data.get('performance_summary', {})
                if perf_summary:
                    report_lines.append("24-HOUR PERFORMANCE SUMMARY")
                    report_lines.append("-" * 30)
                    report_lines.append(f"Starting Balance: ${perf_summary.get('starting_balance', 0):.2f}")
                    report_lines.append(f"Ending Balance: ${perf_summary.get('ending_balance', 0):.2f}")
                    report_lines.append(f"Balance Change: ${perf_summary.get('balance_change', 0):.2f}")
                    report_lines.append(f"Average Risk Score: {perf_summary.get('avg_risk_score', 0):.1f}")
                    report_lines.append(f"Maximum Risk Score: {perf_summary.get('max_risk_score', 0):.1f}")
                    report_lines.append(f"Average Exposure: {perf_summary.get('avg_exposure_pct', 0):.2f}%")
                    report_lines.append(f"Maximum Exposure: {perf_summary.get('max_exposure_pct', 0):.2f}%")

                return "\n".join(report_lines)

            else:
                raise ValueError(f"Unsupported export format: {format}")

        except Exception as e:
            logger.error(f"❌ Failed to export report: {str(e)}")
            return f"Error generating report: {str(e)}"

# Utility functions for dashboard integration
def create_dashboard(trading_engine: EnhancedTradingEngine) -> RiskMonitoringDashboard:
    """Create and initialize monitoring dashboard"""

    dashboard = RiskMonitoringDashboard(
        risk_enforcement=trading_engine.risk_enforcement,
        trading_engine=trading_engine
    )

    return dashboard

def print_dashboard_summary(dashboard: RiskMonitoringDashboard):
    """Print a formatted dashboard summary to console"""

    data = dashboard.get_dashboard_data()
    metrics = data.get('current_metrics', {})

    print("\n" + "=" * 60)
    print("INSTITUTIONAL RISK MONITORING DASHBOARD")
    print("=" * 60)
    print(f"Time: {datetime.fromtimestamp(metrics.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"System Health: {metrics.get('system_health', 'UNKNOWN')}")
    print("-" * 60)
    print(f"Account Balance: ${metrics.get('account_balance', 0):,.2f}")
    print(f"Total Exposure: {metrics.get('total_exposure_pct', 0):.2f}%")
    print(f"Daily PnL: {metrics.get('daily_pnl_pct', 0):.2f}%")
    print(f"Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"Risk Score: {metrics.get('risk_score', 0):.1f}/100")
    print(f"Open Positions: {metrics.get('open_positions', 0)}")
    print(f"Circuit Breaker: {metrics.get('circuit_breaker_state', 'UNKNOWN')}")
    print(f"Trades Executed: {metrics.get('trades_executed_today', 0)}")
    print(f"Success Rate: {metrics.get('success_rate', 0):.1f}%")
    print(f"Total PnL: ${metrics.get('total_pnl', 0):,.2f}")

    alerts = metrics.get('active_alerts', [])
    if alerts:
        print("-" * 60)
        print("ACTIVE ALERTS:")
        for alert in alerts[:5]:  # Show first 5 alerts
            print(f"  {alert}")
        if len(alerts) > 5:
            print(f"  ... and {len(alerts) - 5} more alerts")

    print("=" * 60 + "\n")
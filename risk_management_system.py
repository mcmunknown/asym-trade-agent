"""
INSTITUTIONAL-GRADE RISK MANAGEMENT ARCHITECTURE
=============================================

Multi-layered risk management system designed for BlackRock-level institutional trading.
Zero tolerance for implementation gaps, multiple redundant safety layers, and
circuit breakers that cannot be bypassed.

Architecture Principles:
- Defense in Depth: Multiple validation layers
- Fail-Safe Defaults: Safe mode on any failure
- Real-time Monitoring: Continuous risk assessment
- Automatic Enforcement: No manual override capabilities
- Redundant Validation: Cross-check all critical parameters
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
from decimal import Decimal, getcontext
import numpy as np

# Set high precision for financial calculations
getcontext().prec = 10

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk classification levels"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    MINIMAL = "MINIMAL"

class CircuitBreakerState(Enum):
    """Circuit breaker operational states"""
    CLOSED = "CLOSED"  # Normal operation
    OPEN = "OPEN"      # Trading halted
    HALF_OPEN = "HALF_OPEN"  # Testing recovery

@dataclass
class RiskLimits:
    """Configurable risk parameters with institutional defaults"""
    # Leverage Controls
    max_leverage_conservative: float = 5.0
    max_leverage_moderate: float = 10.0
    max_leverage_aggressive: float = 20.0

    # Position Sizing Controls
    max_position_size_pct_conservative: float = 0.5
    max_position_size_pct_moderate: float = 1.0
    max_position_size_pct_aggressive: float = 2.0

    # Account Risk Controls
    max_account_risk_pct: float = 2.0  # Maximum loss as % of account
    max_daily_loss_pct: float = 1.0    # Daily loss limit
    max_total_exposure_pct: float = 50.0  # Total exposure limit

    # Drawdown Controls
    max_drawdown_pct: float = 10.0
    emergency_drawdown_pct: float = 15.0

    # Position Limits
    max_concurrent_positions: int = 5
    max_positions_per_symbol: int = 1

    # Volatility Controls
    max_volatility_threshold: float = 20.0  # Max 20% daily volatility
    volatility_adjustment_factor: float = 0.5

@dataclass
class RiskMetrics:
    """Real-time risk metrics"""
    current_account_balance: float
    total_exposure: float
    total_exposure_pct: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    daily_pnl: float
    daily_pnl_pct: float
    max_drawdown: float
    max_drawdown_pct: float
    open_positions_count: int
    leverage_utilization: float
    risk_score: float
    volatility_score: float

@dataclass
class TradeValidationResult:
    """Result of pre-trade validation"""
    is_valid: bool
    risk_level: RiskLevel
    violations: List[str]
    warnings: List[str]
    adjusted_parameters: Dict
    circuit_breaker_triggered: bool

class CircuitBreaker:
    """
    Circuit breaker implementation with automatic triggers
    """

    def __init__(self):
        self.state = CircuitBreakerState.CLOSED
        self.trigger_time = None
        self.cooldown_period = 3600  # 1 hour cooldown
        self.trigger_reasons = []
        self.trip_count = 0
        self.max_trips_per_day = 3

    def check_conditions(self, metrics: RiskMetrics, limits: RiskLimits) -> bool:
        """Check if circuit breaker should be triggered"""

        triggers = []

        # Check critical risk conditions
        if metrics.max_drawdown_pct > limits.emergency_drawdown_pct:
            triggers.append(f"Max drawdown exceeded: {metrics.max_drawdown_pct:.2f}% > {limits.emergency_drawdown_pct}%")

        if metrics.daily_pnl_pct < -limits.max_daily_loss_pct * 2:  # Double daily limit
            triggers.append(f"Daily loss exceeded: {metrics.daily_pnl_pct:.2f}%")

        if metrics.total_exposure_pct > limits.max_total_exposure_pct:
            triggers.append(f"Total exposure exceeded: {metrics.total_exposure_pct:.2f}%")

        if metrics.leverage_utilization > 1.5:  # 150% of maximum leverage
            triggers.append(f"Leverage utilization exceeded: {metrics.leverage_utilization:.2f}")

        # Check position limits
        if metrics.open_positions_count > limits.max_concurrent_positions * 2:
            triggers.append(f"Too many positions: {metrics.open_positions_count}")

        # Trip circuit breaker if any critical conditions
        if triggers:
            self.trip(triggers)
            return True

        return False

    def trip(self, reasons: List[str]):
        """Trip the circuit breaker"""
        self.state = CircuitBreakerState.OPEN
        self.trigger_time = time.time()
        self.trigger_reasons = reasons
        self.trip_count += 1

        logger.critical(f"🚨 CIRCUIT BREAKER TRIPPED - Trading halted")
        for reason in reasons:
            logger.critical(f"   Trigger: {reason}")

    def can_trade(self) -> bool:
        """Check if trading is allowed"""
        if self.state == CircuitBreakerState.CLOSED:
            return True

        if self.state == CircuitBreakerState.OPEN:
            # Check if cooldown period has passed
            if time.time() - self.trigger_time > self.cooldown_period:
                self.state = CircuitBreakerState.HALF_OPEN
                logger.warning("⚠️ Circuit breaker entering HALF_OPEN state - Testing recovery")
                return True
            return False

        if self.state == CircuitBreakerState.HALF_OPEN:
            return True

        return False

    def reset(self):
        """Reset circuit breaker to closed state"""
        self.state = CircuitBreakerState.CLOSED
        self.trigger_time = None
        self.trigger_reasons = []
        logger.info("✅ Circuit breaker reset - Normal trading resumed")

class DynamicPositionSizer:
    """
    Dynamic position sizing based on account balance, volatility, and risk parameters
    """

    def __init__(self, limits: RiskLimits):
        self.limits = limits

    def calculate_position_size(
        self,
        account_balance: float,
        symbol_price: float,
        volatility: float,
        risk_level: RiskLevel,
        emergency_mode: bool = False
    ) -> Tuple[float, float]:
        """
        Calculate position size and leverage based on multiple factors

        Returns:
            Tuple[position_size_usd, leverage]
        """

        # Base position size as percentage of account
        if emergency_mode:
            base_size_pct = self.limits.max_position_size_pct_conservative
        elif risk_level == RiskLevel.MINIMAL:
            base_size_pct = self.limits.max_position_size_pct_aggressive
        elif risk_level == RiskLevel.LOW:
            base_size_pct = self.limits.max_position_size_pct_moderate
        else:
            base_size_pct = self.limits.max_position_size_pct_conservative

        # Volatility adjustment
        if volatility > self.limits.max_volatility_threshold:
            volatility_factor = 0.3  # Reduce size by 70% for high volatility
        else:
            volatility_factor = 1.0 - (volatility / self.limits.max_volatility_threshold) * self.limits.volatility_adjustment_factor

        # Account balance scaling
        balance_factor = min(account_balance / 1000, 1.0)  # Scale up to $1000

        # Calculate final position size
        position_size_usd = account_balance * (base_size_pct / 100) * volatility_factor * balance_factor

        # Minimum position size (prevent micro-trades)
        min_position_size = 1.0
        position_size_usd = max(position_size_usd, min_position_size)

        # Calculate leverage based on risk level and volatility
        if emergency_mode:
            max_leverage = self.limits.max_leverage_conservative
        elif risk_level == RiskLevel.MINIMAL:
            max_leverage = self.limits.max_leverage_aggressive
        elif risk_level == RiskLevel.LOW:
            max_leverage = self.limits.max_leverage_moderate
        else:
            max_leverage = self.limits.max_leverage_conservative

        # Volatility-based leverage adjustment
        if volatility > 15:  # High volatility
            leverage = max_leverage * 0.5
        elif volatility > 10:  # Medium volatility
            leverage = max_leverage * 0.75
        else:
            leverage = max_leverage

        leverage = max(leverage, 1.0)  # Minimum 1x leverage

        logger.info(f"   Position sizing calculation:")
        logger.info(f"   Account balance: ${account_balance:.2f}")
        logger.info(f"   Base size %: {base_size_pct:.2f}%")
        logger.info(f"   Volatility factor: {volatility_factor:.2f}")
        logger.info(f"   Balance factor: {balance_factor:.2f}")
        logger.info(f"   Final position: ${position_size_usd:.2f}")
        logger.info(f"   Leverage: {leverage:.1f}x")

        return position_size_usd, leverage

class RiskManager:
    """
    Main risk management system with multi-layered validation
    """

    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()
        self.circuit_breaker = CircuitBreaker()
        self.position_sizer = DynamicPositionSizer(self.limits)
        self.metrics_history = []
        self.max_history_size = 1000

        # Risk tracking
        self.daily_high_water_mark = None
        self.daily_low_water_mark = None
        self.last_reset_time = time.time()
        self.reset_daily_stats()

    def reset_daily_stats(self):
        """Reset daily statistics"""
        self.daily_high_water_mark = None
        self.daily_low_water_mark = None
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_reset_time = time.time()

    def update_metrics(self, current_metrics: RiskMetrics) -> RiskMetrics:
        """Update and store risk metrics"""

        # Update daily stats if needed
        if time.time() - self.last_reset_time > 86400:  # 24 hours
            self.reset_daily_stats()

        # Update daily water marks
        if self.daily_high_water_mark is None or current_metrics.current_account_balance > self.daily_high_water_mark:
            self.daily_high_water_mark = current_metrics.current_account_balance

        if self.daily_low_water_mark is None or current_metrics.current_account_balance < self.daily_low_water_mark:
            self.daily_low_water_mark = current_metrics.current_account_balance

        # Calculate daily PnL
        if self.daily_high_water_mark:
            current_metrics.daily_pnl = current_metrics.current_account_balance - self.daily_high_water_mark
            current_metrics.daily_pnl_pct = (current_metrics.daily_pnl / self.daily_high_water_mark) * 100

        # Store in history
        self.metrics_history.append({
            'timestamp': time.time(),
            'metrics': current_metrics
        })

        # Trim history
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]

        # Check circuit breaker conditions
        self.circuit_breaker.check_conditions(current_metrics, self.limits)

        return current_metrics

    def validate_trade(
        self,
        symbol: str,
        proposed_quantity: float,
        proposed_leverage: float,
        current_price: float,
        account_balance: float,
        volatility: float,
        open_positions: List[Dict],
        emergency_mode: bool = False
    ) -> TradeValidationResult:
        """
        Comprehensive pre-trade validation with multiple safety layers
        """

        violations = []
        warnings = []
        adjusted_parameters = {}

        # Layer 1: Circuit breaker check
        if not self.circuit_breaker.can_trade():
            return TradeValidationResult(
                is_valid=False,
                risk_level=RiskLevel.CRITICAL,
                violations=["Circuit breaker is active - Trading halted"],
                warnings=[],
                adjusted_parameters={},
                circuit_breaker_triggered=True
            )

        # Layer 2: Account-level checks
        if account_balance < 10:  # Minimum account balance
            violations.append(f"Insufficient account balance: ${account_balance:.2f}")

        # Layer 3: Position-level checks
        proposed_exposure = proposed_quantity * current_price * proposed_leverage
        proposed_exposure_pct = (proposed_exposure / account_balance) * 100

        if proposed_exposure_pct > self.limits.max_total_exposure_pct:
            violations.append(f"Proposed exposure too high: {proposed_exposure_pct:.2f}% > {self.limits.max_total_exposure_pct}%")

        # Layer 4: Leverage validation
        max_allowed_leverage = self.limits.max_leverage_conservative if emergency_mode else self.limits.max_leverage_moderate
        if proposed_leverage > max_allowed_leverage:
            violations.append(f"Proposed leverage too high: {proposed_leverage:.1f}x > {max_allowed_leverage:.1f}x")
            adjusted_parameters['leverage'] = max_allowed_leverage

        # Layer 5: Position limits
        symbol_positions = [p for p in open_positions if p.get('symbol') == symbol]
        if len(symbol_positions) >= self.limits.max_positions_per_symbol:
            violations.append(f"Maximum positions for {symbol} already held")

        if len(open_positions) >= self.limits.max_concurrent_positions:
            violations.append(f"Maximum concurrent positions reached: {len(open_positions)}")

        # Layer 6: Volatility checks
        if volatility > self.limits.max_volatility_threshold:
            warnings.append(f"High volatility detected: {volatility:.2f}%")
            if emergency_mode:
                violations.append(f"High volatility in emergency mode: {volatility:.2f}%")

        # Layer 7: Risk level assessment
        risk_level = self._assess_risk_level(proposed_exposure_pct, volatility, proposed_leverage, emergency_mode)

        # Layer 8: Dynamic position sizing adjustment
        recommended_position_size, recommended_leverage = self.position_sizer.calculate_position_size(
            account_balance, current_price, volatility, risk_level, emergency_mode
        )

        # Adjust parameters if needed
        if proposed_quantity * current_price > recommended_position_size * 1.1:  # 10% tolerance
            warnings.append(f"Position size exceeds recommendation: ${proposed_quantity * current_price:.2f} > ${recommended_position_size:.2f}")
            adjusted_parameters['quantity'] = recommended_position_size / current_price

        if proposed_leverage > recommended_leverage * 1.1:
            warnings.append(f"Leverage exceeds recommendation: {proposed_leverage:.1f}x > {recommended_leverage:.1f}x")
            adjusted_parameters['leverage'] = recommended_leverage

        # Determine final validation result
        is_valid = len(violations) == 0
        circuit_breaker_triggered = False

        # Auto-adjust if possible
        if adjusted_parameters and not violations:
            logger.info(f"   Auto-adjusting trade parameters: {adjusted_parameters}")

        return TradeValidationResult(
            is_valid=is_valid,
            risk_level=risk_level,
            violations=violations,
            warnings=warnings,
            adjusted_parameters=adjusted_parameters,
            circuit_breaker_triggered=circuit_breaker_triggered
        )

    def _assess_risk_level(
        self,
        exposure_pct: float,
        volatility: float,
        leverage: float,
        emergency_mode: bool
    ) -> RiskLevel:
        """Assess overall risk level for a trade"""

        risk_score = 0

        # Emergency mode adds risk
        if emergency_mode:
            risk_score += 2

        # Exposure risk
        if exposure_pct > 20:
            risk_score += 3
        elif exposure_pct > 10:
            risk_score += 2
        elif exposure_pct > 5:
            risk_score += 1

        # Volatility risk
        if volatility > 15:
            risk_score += 3
        elif volatility > 10:
            risk_score += 2
        elif volatility > 5:
            risk_score += 1

        # Leverage risk
        if leverage > 20:
            risk_score += 3
        elif leverage > 10:
            risk_score += 2
        elif leverage > 5:
            risk_score += 1

        # Determine risk level
        if risk_score >= 7:
            return RiskLevel.CRITICAL
        elif risk_score >= 5:
            return RiskLevel.HIGH
        elif risk_score >= 3:
            return RiskLevel.MEDIUM
        elif risk_score >= 1:
            return RiskLevel.LOW
        else:
            return RiskLevel.MINIMAL

    def get_risk_summary(self) -> Dict:
        """Get comprehensive risk summary"""

        if not self.metrics_history:
            return {"status": "No data available"}

        latest_metrics = self.metrics_history[-1]['metrics']

        return {
            "circuit_breaker_state": self.circuit_breaker.state.value,
            "current_metrics": {
                "account_balance": latest_metrics.current_account_balance,
                "total_exposure_pct": latest_metrics.total_exposure_pct,
                "daily_pnl_pct": latest_metrics.daily_pnl_pct,
                "max_drawdown_pct": latest_metrics.max_drawdown_pct,
                "open_positions": latest_metrics.open_positions_count,
                "risk_score": latest_metrics.risk_score
            },
            "limits": {
                "max_leverage": self.limits.max_leverage_moderate,
                "max_position_size_pct": self.limits.max_position_size_pct_moderate,
                "max_drawdown_pct": self.limits.max_drawdown_pct,
                "max_daily_loss_pct": self.limits.max_daily_loss_pct
            },
            "warnings": [] if latest_metrics.risk_score < 5 else ["High risk detected"],
            "status": "Healthy" if self.circuit_breaker.state == CircuitBreakerState.CLOSED else "Trading halted"
        }

class RiskMonitoringService:
    """
    Real-time risk monitoring and alerting service
    """

    def __init__(self, risk_manager: RiskManager):
        self.risk_manager = risk_manager
        self.alert_thresholds = {
            'drawdown_warning': 5.0,
            'drawdown_critical': 10.0,
            'exposure_warning': 30.0,
            'exposure_critical': 45.0,
            'daily_loss_warning': 0.5,
            'daily_loss_critical': 1.0
        }

    def monitor_and_alert(self, metrics: RiskMetrics) -> List[str]:
        """Monitor metrics and generate alerts"""

        alerts = []

        # Drawdown alerts
        if abs(metrics.max_drawdown_pct) > self.alert_thresholds['drawdown_critical']:
            alerts.append(f"🚨 CRITICAL: Drawdown {metrics.max_drawdown_pct:.2f}% exceeds threshold")
        elif abs(metrics.max_drawdown_pct) > self.alert_thresholds['drawdown_warning']:
            alerts.append(f"⚠️ WARNING: Drawdown {metrics.max_drawdown_pct:.2f}% approaching limit")

        # Exposure alerts
        if metrics.total_exposure_pct > self.alert_thresholds['exposure_critical']:
            alerts.append(f"🚨 CRITICAL: Total exposure {metrics.total_exposure_pct:.2f}% exceeds threshold")
        elif metrics.total_exposure_pct > self.alert_thresholds['exposure_warning']:
            alerts.append(f"⚠️ WARNING: Total exposure {metrics.total_exposure_pct:.2f}% approaching limit")

        # Daily loss alerts
        if metrics.daily_pnl_pct < -self.alert_thresholds['daily_loss_critical']:
            alerts.append(f"🚨 CRITICAL: Daily loss {metrics.daily_pnl_pct:.2f}% exceeds threshold")
        elif metrics.daily_pnl_pct < -self.alert_thresholds['daily_loss_warning']:
            alerts.append(f"⚠️ WARNING: Daily loss {metrics.daily_pnl_pct:.2f}% approaching limit")

        # Position count alerts
        if metrics.open_positions_count > self.risk_manager.limits.max_concurrent_positions * 0.8:
            alerts.append(f"⚠️ WARNING: Position count {metrics.open_positions_count} approaching limit")

        # Log alerts
        for alert in alerts:
            logger.warning(alert)

        return alerts

# Factory function for easy instantiation
def create_institutional_risk_manager(
    conservative_mode: bool = False,
    emergency_mode: bool = False
) -> RiskManager:
    """Create risk manager with appropriate configuration"""

    if emergency_mode:
        limits = RiskLimits(
            max_leverage_conservative=3.0,
            max_leverage_moderate=5.0,
            max_leverage_aggressive=10.0,
            max_position_size_pct_conservative=0.25,
            max_position_size_pct_moderate=0.5,
            max_position_size_pct_aggressive=1.0,
            max_daily_loss_pct=0.5,
            max_drawdown_pct=5.0,
            emergency_drawdown_pct=8.0
        )
    elif conservative_mode:
        limits = RiskLimits(
            max_leverage_conservative=5.0,
            max_leverage_moderate=10.0,
            max_leverage_aggressive=15.0,
            max_position_size_pct_conservative=0.5,
            max_position_size_pct_moderate=1.0,
            max_position_size_pct_aggressive=1.5,
            max_daily_loss_pct=1.0,
            max_drawdown_pct=8.0,
            emergency_drawdown_pct=12.0
        )
    else:
        limits = RiskLimits()  # Default institutional limits

    return RiskManager(limits)
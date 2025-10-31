"""
RISK ENFORCEMENT LAYER
=====================

This layer provides the integration between the risk management system and the trading engine.
It enforces all risk controls and cannot be bypassed by any trading logic.

Key Features:
- Mandatory pre-trade validation
- Real-time risk monitoring
- Automatic position sizing adjustment
- Circuit breaker integration
- Emergency mode enforcement
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time

from risk_management_system import (
    RiskManager, RiskMetrics, TradeValidationResult, RiskLevel,
    create_institutional_risk_manager, RiskMonitoringService
)

logger = logging.getLogger(__name__)

@dataclass
class TradeRequest:
    """Standardized trade request format"""
    symbol: str
    direction: str  # 'Buy' or 'Sell'
    entry_price: float
    quantity: float
    leverage: float
    stop_loss_price: float
    take_profit_price: float
    signal_confidence: float
    volatility: float
    emergency_mode: bool = False

@dataclass
class TradeExecutionPlan:
    """Validated and adjusted trade execution plan"""
    symbol: str
    direction: str
    entry_price: float
    quantity: float
    leverage: float
    stop_loss_price: float
    take_profit_price: float
    position_size_usd: float
    exposure_usd: float
    risk_level: RiskLevel
    validation_result: TradeValidationResult
    execution_allowed: bool

class RiskEnforcementLayer:
    """
    Critical enforcement layer that integrates risk controls into the trading engine
    """

    def __init__(self, config):
        self.config = config

        # Initialize risk manager with current settings
        self.risk_manager = create_institutional_risk_manager(
            conservative_mode=True,  # Always use conservative mode for safety
            emergency_mode=config.EMERGENCY_DEEPSEEK_ONLY
        )

        # Initialize monitoring service
        self.monitoring_service = RiskMonitoringService(self.risk_manager)

        # Cache for account balance
        self._last_balance_update = 0
        self._cached_balance = None
        self.balance_cache_ttl = 60  # 1 minute cache

        # Trading state tracking
        self.active_positions = {}
        self.daily_trades = 0
        self.last_trade_time = 0

        logger.info("✅ Risk enforcement layer initialized")
        logger.info(f"   Emergency mode: {config.EMERGENCY_DEEPSEEK_ONLY}")
        logger.info(f"   Conservative mode: True")
        logger.info(f"   Max leverage: {self.risk_manager.limits.max_leverage_moderate}x")

    def get_account_balance(self, bybit_client) -> float:
        """Get current account balance with caching"""

        current_time = time.time()

        # Use cached balance if still valid
        if (self._cached_balance is not None and
            current_time - self._last_balance_update < self.balance_cache_ttl):
            return self._cached_balance

        try:
            # Fetch fresh balance from Bybit
            balance_info = bybit_client.get_wallet_balance()

            if balance_info and 'result' in balance_info:
                # Get USDT balance
                usdt_balance = 0.0
                for coin in balance_info['result'].get('list', []):
                    if coin.get('coin') == 'USDT':
                        usdt_balance = float(coin.get('walletBalance', 0))
                        break

                self._cached_balance = usdt_balance
                self._last_balance_update = current_time

                logger.debug(f"Updated account balance: ${usdt_balance:.2f}")
                return usdt_balance

        except Exception as e:
            logger.error(f"Failed to fetch account balance: {str(e)}")

            # Use cached balance if available, even if expired
            if self._cached_balance is not None:
                logger.warning(f"Using cached balance: ${self._cached_balance:.2f}")
                return self._cached_balance

        # Fallback to default minimum balance
        logger.error("Unable to fetch account balance, using minimum balance")
        return 100.0  # Minimum balance for safety

    def update_risk_metrics(self, bybit_client) -> RiskMetrics:
        """Update current risk metrics from live data"""

        try:
            # Get current account balance
            account_balance = self.get_account_balance(bybit_client)

            # Get current positions
            positions_info = bybit_client.get_positions()

            total_exposure = 0.0
            unrealized_pnl = 0.0
            open_positions_count = 0
            leverage_utilization = 0.0

            if positions_info and 'result' in positions_info:
                for position in positions_info['result'].get('list', []):
                    size = float(position.get('size', 0))
                    if size != 0:  # Active position
                        open_positions_count += 1
                        position_value = float(position.get('positionValue', 0))
                        position_leverage = float(position.get('leverage', 1))

                        total_exposure += abs(position_value)
                        unrealized_pnl += float(position.get('unrealisedPnl', 0))
                        leverage_utilization = max(leverage_utilization, position_leverage)

            # Calculate percentages
            total_exposure_pct = (total_exposure / account_balance * 100) if account_balance > 0 else 0
            unrealized_pnl_pct = (unrealized_pnl / account_balance * 100) if account_balance > 0 else 0

            # Get daily PnL from metrics history
            daily_pnl = 0.0
            daily_pnl_pct = 0.0
            max_drawdown = 0.0
            max_drawdown_pct = 0.0

            if self.risk_manager.metrics_history:
                # Get start of day balance
                start_of_day_balance = account_balance - unrealized_pnl
                daily_pnl = unrealized_pnl
                daily_pnl_pct = (daily_pnl / start_of_day_balance * 100) if start_of_day_balance > 0 else 0

                # Calculate drawdown
                if self.risk_manager.daily_high_water_mark:
                    drawdown = self.risk_manager.daily_high_water_mark - account_balance
                    max_drawdown = max(drawdown, max_drawdown)
                    max_drawdown_pct = (max_drawdown / self.risk_manager.daily_high_water_mark * 100) if self.risk_manager.daily_high_water_mark > 0 else 0

            # Calculate risk score
            risk_score = self._calculate_risk_score(
                total_exposure_pct, unrealized_pnl_pct, leverage_utilization, open_positions_count
            )

            # Calculate volatility score (simplified)
            volatility_score = 10.0  # Default moderate volatility

            # Create metrics object
            metrics = RiskMetrics(
                current_account_balance=account_balance,
                total_exposure=total_exposure,
                total_exposure_pct=total_exposure_pct,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=unrealized_pnl_pct,
                daily_pnl=daily_pnl,
                daily_pnl_pct=daily_pnl_pct,
                max_drawdown=max_drawdown,
                max_drawdown_pct=max_drawdown_pct,
                open_positions_count=open_positions_count,
                leverage_utilization=leverage_utilization,
                risk_score=risk_score,
                volatility_score=volatility_score
            )

            # Update risk manager metrics
            updated_metrics = self.risk_manager.update_metrics(metrics)

            # Monitor and generate alerts
            alerts = self.monitoring_service.monitor_and_alert(updated_metrics)

            # Log current status
            if self.daily_trades % 10 == 0 or alerts:  # Log every 10 trades or on alerts
                logger.info(f"📊 Risk Metrics Update:")
                logger.info(f"   Balance: ${account_balance:.2f}")
                logger.info(f"   Exposure: {total_exposure_pct:.2f}% (${total_exposure:.2f})")
                logger.info(f"   Daily PnL: {daily_pnl_pct:.2f}%")
                logger.info(f"   Max Drawdown: {max_drawdown_pct:.2f}%")
                logger.info(f"   Open Positions: {open_positions_count}")
                logger.info(f"   Risk Score: {risk_score:.1f}")

                if alerts:
                    logger.warning(f"   Alerts: {len(alerts)} active")

            return updated_metrics

        except Exception as e:
            logger.error(f"Failed to update risk metrics: {str(e)}")
            # Return safe default metrics
            return RiskMetrics(
                current_account_balance=100.0,
                total_exposure=0.0,
                total_exposure_pct=0.0,
                unrealized_pnl=0.0,
                unrealized_pnl_pct=0.0,
                daily_pnl=0.0,
                daily_pnl_pct=0.0,
                max_drawdown=0.0,
                max_drawdown_pct=0.0,
                open_positions_count=0,
                leverage_utilization=0.0,
                risk_score=0.0,
                volatility_score=0.0
            )

    def _calculate_risk_score(
        self,
        exposure_pct: float,
        pnl_pct: float,
        leverage_utilization: float,
        position_count: int
    ) -> float:
        """Calculate overall risk score (0-100)"""

        score = 0.0

        # Exposure risk (0-30 points)
        if exposure_pct > 50:
            score += 30
        elif exposure_pct > 30:
            score += 20
        elif exposure_pct > 20:
            score += 10
        elif exposure_pct > 10:
            score += 5

        # PnL risk (0-25 points)
        if pnl_pct < -10:
            score += 25
        elif pnl_pct < -5:
            score += 15
        elif pnl_pct < -2:
            score += 10
        elif pnl_pct < -1:
            score += 5

        # Leverage risk (0-25 points)
        if leverage_utilization > 20:
            score += 25
        elif leverage_utilization > 15:
            score += 20
        elif leverage_utilization > 10:
            score += 15
        elif leverage_utilization > 5:
            score += 10
        elif leverage_utilization > 2:
            score += 5

        # Position count risk (0-20 points)
        if position_count > 5:
            score += 20
        elif position_count > 3:
            score += 10
        elif position_count > 1:
            score += 5

        return min(score, 100.0)

    def validate_and_adjust_trade(
        self,
        trade_request: TradeRequest,
        bybit_client
    ) -> TradeExecutionPlan:
        """
        Validate trade request and adjust parameters according to risk rules
        """

        logger.info(f"🔍 Risk validation for {trade_request.symbol}")
        logger.info(f"   Requested quantity: {trade_request.quantity}")
        logger.info(f"   Requested leverage: {trade_request.leverage}x")
        logger.info(f"   Entry price: ${trade_request.entry_price:.4f}")

        # Update current risk metrics
        current_metrics = self.update_risk_metrics(bybit_client)

        # Get current open positions
        positions_info = bybit_client.get_positions()
        open_positions = []

        if positions_info and 'result' in positions_info:
            for position in positions_info['result'].get('list', []):
                size = float(position.get('size', 0))
                if size != 0:
                    open_positions.append({
                        'symbol': position.get('symbol'),
                        'size': size,
                        'side': position.get('side'),
                        'leverage': float(position.get('leverage', 1)),
                        'value': float(position.get('positionValue', 0))
                    })

        # Perform comprehensive trade validation
        validation_result = self.risk_manager.validate_trade(
            symbol=trade_request.symbol,
            proposed_quantity=trade_request.quantity,
            proposed_leverage=trade_request.leverage,
            current_price=trade_request.entry_price,
            account_balance=current_metrics.current_account_balance,
            volatility=trade_request.volatility,
            open_positions=open_positions,
            emergency_mode=trade_request.emergency_mode
        )

        # Apply adjustments if validation passed
        final_quantity = trade_request.quantity
        final_leverage = trade_request.leverage

        if validation_result.adjusted_parameters:
            if 'quantity' in validation_result.adjusted_parameters:
                final_quantity = validation_result.adjusted_parameters['quantity']
                logger.info(f"   ✅ Quantity adjusted to: {final_quantity}")

            if 'leverage' in validation_result.adjusted_parameters:
                final_leverage = validation_result.adjusted_parameters['leverage']
                logger.info(f"   ✅ Leverage adjusted to: {final_leverage}x")

        # Calculate final position metrics
        position_size_usd = final_quantity * trade_request.entry_price
        exposure_usd = position_size_usd * final_leverage

        # Log validation results
        logger.info(f"   Risk level: {validation_result.risk_level.value}")
        logger.info(f"   Position size: ${position_size_usd:.2f}")
        logger.info(f"   Total exposure: ${exposure_usd:.2f}")
        logger.info(f"   Exposure % of account: {(exposure_usd / current_metrics.current_account_balance * 100):.2f}%")

        if validation_result.violations:
            logger.error(f"   ❌ VIOLATIONS: {len(validation_result.violations)}")
            for violation in validation_result.violations:
                logger.error(f"      - {violation}")

        if validation_result.warnings:
            logger.warning(f"   ⚠️ WARNINGS: {len(validation_result.warnings)}")
            for warning in validation_result.warnings:
                logger.warning(f"      - {warning}")

        if validation_result.circuit_breaker_triggered:
            logger.error("   🚨 CIRCUIT BREAKER TRIGGERED - Trade rejected")

        # Create execution plan
        execution_plan = TradeExecutionPlan(
            symbol=trade_request.symbol,
            direction=trade_request.direction,
            entry_price=trade_request.entry_price,
            quantity=final_quantity,
            leverage=final_leverage,
            stop_loss_price=trade_request.stop_loss_price,
            take_profit_price=trade_request.take_profit_price,
            position_size_usd=position_size_usd,
            exposure_usd=exposure_usd,
            risk_level=validation_result.risk_level,
            validation_result=validation_result,
            execution_allowed=validation_result.is_valid and not validation_result.circuit_breaker_triggered
        )

        # Update daily trade counter
        if execution_plan.execution_allowed:
            self.daily_trades += 1
            self.last_trade_time = time.time()

        return execution_plan

    def pre_trade_check(self, symbol: str, bybit_client) -> bool:
        """
        Quick pre-trade check to verify trading is allowed
        """

        # Check circuit breaker
        if not self.risk_manager.circuit_breaker.can_trade():
            logger.warning(f"❌ Trading blocked by circuit breaker for {symbol}")
            return False

        # Update metrics quickly
        metrics = self.update_risk_metrics(bybit_client)

        # Check critical thresholds
        if metrics.max_drawdown_pct > self.risk_manager.limits.emergency_drawdown_pct:
            logger.warning(f"❌ Trading blocked - Max drawdown exceeded: {metrics.max_drawdown_pct:.2f}%")
            return False

        if metrics.daily_pnl_pct < -self.risk_manager.limits.max_daily_loss_pct * 2:
            logger.warning(f"❌ Trading blocked - Daily loss exceeded: {metrics.daily_pnl_pct:.2f}%")
            return False

        if metrics.total_exposure_pct > self.risk_manager.limits.max_total_exposure_pct:
            logger.warning(f"❌ Trading blocked - Total exposure exceeded: {metrics.total_exposure_pct:.2f}%")
            return False

        return True

    def post_trade_monitoring(self, symbol: str, execution_result: Dict, bybit_client):
        """
        Monitor trade after execution for risk compliance
        """

        try:
            # Update metrics after trade
            metrics = self.update_risk_metrics(bybit_client)

            # Check if trade caused any risk violations
            if metrics.risk_score > 80:
                logger.critical(f"🚨 High risk score after {symbol} trade: {metrics.risk_score:.1f}")

                # Consider circuit breaker if extremely high risk
                if metrics.risk_score > 95:
                    logger.critical(f"🚨 EMERGENCY: Triggering circuit breaker due to extreme risk")
                    self.risk_manager.circuit_breaker.trip([f"Risk score {metrics.risk_score:.1f} > 95"])

            # Log trade summary
            if execution_result.get('success', False):
                logger.info(f"✅ Trade executed: {symbol}")
                logger.info(f"   Result: {execution_result}")
                logger.info(f"   New account balance: ${metrics.current_account_balance:.2f}")
                logger.info(f"   Total exposure: {metrics.total_exposure_pct:.2f}%")
            else:
                logger.error(f"❌ Trade failed: {symbol}")
                logger.error(f"   Error: {execution_result.get('error', 'Unknown error')}")

        except Exception as e:
            logger.error(f"Post-trade monitoring failed for {symbol}: {str(e)}")

    def get_risk_status(self) -> Dict:
        """Get current risk status summary"""

        risk_summary = self.risk_manager.get_risk_summary()

        # Add additional status information
        risk_summary.update({
            'daily_trades': self.daily_trades,
            'last_trade_time': self.last_trade_time,
            'balance_cache_age': time.time() - self._last_balance_update,
            'enforcement_layer_status': 'Active' if self.risk_manager.circuit_breaker.can_trade() else 'Blocked'
        })

        return risk_summary

    def emergency_stop(self, reason: str):
        """
        Emergency stop - immediately halt all trading
        """

        logger.critical(f"🚨 EMERGENCY STOP ACTIVATED: {reason}")
        self.risk_manager.circuit_breaker.trip([f"Emergency stop: {reason}"])

        # Clear cached balance to force refresh
        self._cached_balance = None
        self._last_balance_update = 0

    def reset_circuit_breaker(self, admin_code: str) -> bool:
        """
        Reset circuit breaker (requires admin authorization)
        """

        # In production, this would require proper authentication
        if admin_code == "RESET_CONFIRMED_2024":
            self.risk_manager.circuit_breaker.reset()
            logger.info("✅ Circuit breaker reset by administrator")
            return True
        else:
            logger.error("❌ Invalid admin code for circuit breaker reset")
            return False
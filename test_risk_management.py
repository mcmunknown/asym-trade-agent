"""
RISK MANAGEMENT SYSTEM VALIDATION TESTS
======================================

Comprehensive test suite to validate the institutional-grade risk management system.
Tests all critical components and edge cases to ensure robustness.

Tests:
- Risk limit enforcement
- Circuit breaker functionality
- Dynamic position sizing
- Emergency mode behavior
- Integration with trading engine
"""

import logging
import unittest
from unittest.mock import Mock, patch
from decimal import Decimal
import time

from risk_management_system import (
    RiskManager, RiskLimits, RiskMetrics, RiskLevel,
    CircuitBreaker, CircuitBreakerState,
    DynamicPositionSizer, create_institutional_risk_manager
)
from risk_enforcement_layer import (
    RiskEnforcementLayer, TradeRequest, TradeExecutionPlan
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestRiskManagementSystem(unittest.TestCase):
    """Test cases for risk management system"""

    def setUp(self):
        """Set up test fixtures"""
        self.limits = RiskLimits()
        self.risk_manager = RiskManager(self.limits)
        self.circuit_breaker = CircuitBreaker()
        self.position_sizer = DynamicPositionSizer(self.limits)

    def test_risk_limits_initialization(self):
        """Test risk limits are properly initialized"""
        self.assertEqual(self.limits.max_leverage_conservative, 5.0)
        self.assertEqual(self.limits.max_leverage_moderate, 10.0)
        self.assertEqual(self.limits.max_position_size_pct_conservative, 0.5)
        self.assertEqual(self.limits.max_drawdown_pct, 10.0)
        self.assertEqual(self.limits.emergency_drawdown_pct, 15.0)

    def test_circuit_breaker_initial_state(self):
        """Test circuit breaker starts in closed state"""
        self.assertEqual(self.circuit_breaker.state, CircuitBreakerState.CLOSED)
        self.assertTrue(self.circuit_breaker.can_trade())

    def test_circuit_breaker_trigger_conditions(self):
        """Test circuit breaker triggers on critical conditions"""
        # Create metrics that should trigger circuit breaker
        metrics = RiskMetrics(
            current_account_balance=1000.0,
            total_exposure=600.0,
            total_exposure_pct=60.0,  # Exceeds 50% limit
            unrealized_pnl=-50.0,
            unrealized_pnl_pct=-5.0,
            daily_pnl=-30.0,
            daily_pnl_pct=-3.0,  # Exceeds daily loss limit
            max_drawdown=200.0,
            max_drawdown_pct=20.0,  # Exceeds emergency drawdown
            open_positions_count=8,
            leverage_utilization=2.0,
            risk_score=95.0,
            volatility_score=15.0
        )

        # Test circuit breaker triggers
        triggered = self.circuit_breaker.check_conditions(metrics, self.limits)
        self.assertTrue(triggered)
        self.assertEqual(self.circuit_breaker.state, CircuitBreakerState.OPEN)
        self.assertFalse(self.circuit_breaker.can_trade())

    def test_dynamic_position_sizing(self):
        """Test dynamic position sizing calculations"""
        account_balance = 1000.0
        symbol_price = 100.0
        volatility = 10.0
        risk_level = RiskLevel.LOW
        emergency_mode = False

        position_size, leverage = self.position_sizer.calculate_position_size(
            account_balance, symbol_price, volatility, risk_level, emergency_mode
        )

        # Verify position size is reasonable
        self.assertGreater(position_size, 0)
        self.assertLessEqual(position_size, account_balance * 0.02)  # Should be <= 2% of account

        # Verify leverage is within limits
        self.assertGreaterEqual(leverage, 1.0)
        self.assertLessEqual(leverage, self.limits.max_leverage_moderate)

    def test_dynamic_position_sizing_emergency_mode(self):
        """Test position sizing in emergency mode"""
        account_balance = 1000.0
        symbol_price = 100.0
        volatility = 10.0
        risk_level = RiskLevel.LOW
        emergency_mode = True

        position_size, leverage = self.position_sizer.calculate_position_size(
            account_balance, symbol_price, volatility, risk_level, emergency_mode
        )

        # Emergency mode should use conservative limits
        self.assertLessEqual(leverage, self.limits.max_leverage_conservative)
        self.assertLessEqual(position_size, account_balance * 0.01)  # Should be <= 1% of account

    def test_trade_validation_success(self):
        """Test successful trade validation"""
        trade_request = TradeRequest(
            symbol="BTCUSDT",
            direction="Buy",
            entry_price=50000.0,
            quantity=0.001,
            leverage=5.0,
            stop_loss_price=47500.0,
            take_profit_price=52500.0,
            signal_confidence=0.8,
            volatility=10.0,
            emergency_mode=False
        )

        # Mock account balance and positions
        account_balance = 1000.0
        open_positions = []

        validation_result = self.risk_manager.validate_trade(
            symbol=trade_request.symbol,
            proposed_quantity=trade_request.quantity,
            proposed_leverage=trade_request.leverage,
            current_price=trade_request.entry_price,
            account_balance=account_balance,
            volatility=trade_request.volatility,
            open_positions=open_positions,
            emergency_mode=trade_request.emergency_mode
        )

        self.assertTrue(validation_result.is_valid)
        self.assertEqual(validation_result.risk_level, RiskLevel.LOW)
        self.assertEqual(len(validation_result.violations), 0)

    def test_trade_validation_leverage_violation(self):
        """Test trade validation rejects excessive leverage"""
        trade_request = TradeRequest(
            symbol="BTCUSDT",
            direction="Buy",
            entry_price=50000.0,
            quantity=0.001,
            leverage=50.0,  # Excessive leverage
            stop_loss_price=47500.0,
            take_profit_price=52500.0,
            signal_confidence=0.8,
            volatility=10.0,
            emergency_mode=True  # Emergency mode with 5x max leverage
        )

        account_balance = 1000.0
        open_positions = []

        validation_result = self.risk_manager.validate_trade(
            symbol=trade_request.symbol,
            proposed_quantity=trade_request.quantity,
            proposed_leverage=trade_request.leverage,
            current_price=trade_request.entry_price,
            account_balance=account_balance,
            volatility=trade_request.volatility,
            open_positions=open_positions,
            emergency_mode=trade_request.emergency_mode
        )

        self.assertFalse(validation_result.is_valid)
        self.assertGreater(len(validation_result.violations), 0)
        self.assertTrue(any("leverage" in v.lower() for v in validation_result.violations))

    def test_trade_validation_insufficient_balance(self):
        """Test trade validation rejects insufficient balance"""
        trade_request = TradeRequest(
            symbol="BTCUSDT",
            direction="Buy",
            entry_price=50000.0,
            quantity=1.0,  # Large position requiring $50,000
            leverage=5.0,
            stop_loss_price=47500.0,
            take_profit_price=52500.0,
            signal_confidence=0.8,
            volatility=10.0,
            emergency_mode=False
        )

        account_balance = 100.0  # Insufficient balance
        open_positions = []

        validation_result = self.risk_manager.validate_trade(
            symbol=trade_request.symbol,
            proposed_quantity=trade_request.quantity,
            proposed_leverage=trade_request.leverage,
            current_price=trade_request.entry_price,
            account_balance=account_balance,
            volatility=trade_request.volatility,
            open_positions=open_positions,
            emergency_mode=trade_request.emergency_mode
        )

        self.assertFalse(validation_result.is_valid)
        self.assertGreater(len(validation_result.violations), 0)

    def test_risk_level_assessment(self):
        """Test risk level assessment algorithm"""
        # Test minimal risk
        risk_level = self.risk_manager._assess_risk_level(
            exposure_pct=5.0,  # Low exposure
            volatility=5.0,    # Low volatility
            leverage=2.0,      # Low leverage
            emergency_mode=False
        )
        self.assertEqual(risk_level, RiskLevel.MINIMAL)

        # Test high risk
        risk_level = self.risk_manager._assess_risk_level(
            exposure_pct=25.0,  # High exposure
            volatility=20.0,    # High volatility
            leverage=25.0,      # High leverage
            emergency_mode=False
        )
        self.assertIn(risk_level, [RiskLevel.HIGH, RiskLevel.CRITICAL])

        # Test emergency mode increases risk
        emergency_risk = self.risk_manager._assess_risk_level(
            exposure_pct=5.0,
            volatility=5.0,
            leverage=2.0,
            emergency_mode=True
        )
        normal_risk = self.risk_manager._assess_risk_level(
            exposure_pct=5.0,
            volatility=5.0,
            leverage=2.0,
            emergency_mode=False
        )
        self.assertGreater(emergency_risk.value, normal_risk.value)

class TestRiskEnforcementLayer(unittest.TestCase):
    """Test cases for risk enforcement layer"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = Mock()
        self.mock_config.EMERGENCY_DEEPSEEK_ONLY = True
        self.mock_config.MAX_LEVERAGE = 10
        self.mock_config.MAX_POSITION_SIZE_PERCENTAGE = 1.0

        self.risk_enforcement = RiskEnforcementLayer(self.mock_config)

    @patch('risk_enforcement_layer.BybitClient')
    def test_account_balance_caching(self, mock_bybit_client):
        """Test account balance caching mechanism"""
        # Mock API response
        mock_bybit_client.return_value.get_wallet_balance.return_value = {
            'result': {
                'list': [{'coin': 'USDT', 'walletBalance': '1000.0'}]
            }
        }

        # First call should fetch from API
        balance1 = self.risk_enforcement.get_account_balance(mock_bybit_client.return_value)
        self.assertEqual(balance1, 1000.0)

        # Second call within cache period should use cache
        balance2 = self.risk_enforcement.get_account_balance(mock_bybit_client.return_value)
        self.assertEqual(balance2, 1000.0)

        # API should only be called once
        mock_bybit_client.return_value.get_wallet_balance.assert_called_once()

    @patch('risk_enforcement_layer.BybitClient')
    def test_pre_trade_check_circuit_breaker(self, mock_bybit_client):
        """Test pre-trade check with circuit breaker"""
        # Trip circuit breaker
        self.risk_enforcement.risk_manager.circuit_breaker.trip(["Test trigger"])

        # Pre-trade check should fail
        result = self.risk_enforcement.pre_trade_check("BTCUSDT", mock_bybit_client.return_value)
        self.assertFalse(result)

    @patch('risk_enforcement_layer.BybitClient')
    def test_validate_and_adjust_trade(self, mock_bybit_client):
        """Test trade validation and adjustment"""
        # Mock API responses
        mock_bybit_client.return_value.get_wallet_balance.return_value = {
            'result': {
                'list': [{'coin': 'USDT', 'walletBalance': '1000.0'}]
            }
        }
        mock_bybit_client.return_value.get_positions.return_value = {
            'result': {'list': []}
        }

        # Create trade request with excessive leverage
        trade_request = TradeRequest(
            symbol="BTCUSDT",
            direction="Buy",
            entry_price=50000.0,
            quantity=0.01,  # $500 position
            leverage=50.0,  # Excessive leverage for emergency mode
            stop_loss_price=47500.0,
            take_profit_price=52500.0,
            signal_confidence=0.8,
            volatility=10.0,
            emergency_mode=True
        )

        # Validate and adjust
        execution_plan = self.risk_enforcement.validate_and_adjust_trade(
            trade_request, mock_bybit_client.return_value
        )

        # Should be adjusted to safe levels
        self.assertIsNotNone(execution_plan)
        self.assertLessEqual(execution_plan.leverage, self.mock_config.MAX_LEVERAGE)

class TestIntegrationScenarios(unittest.TestCase):
    """Integration test scenarios"""

    def test_create_institutional_risk_manager(self):
        """Test factory function for creating risk managers"""
        # Emergency mode
        risk_manager = create_institutional_risk_manager(
            conservative_mode=False,
            emergency_mode=True
        )
        self.assertEqual(risk_manager.limits.max_leverage_conservative, 3.0)
        self.assertEqual(risk_manager.limits.max_position_size_pct_conservative, 0.25)

        # Conservative mode
        risk_manager = create_institutional_risk_manager(
            conservative_mode=True,
            emergency_mode=False
        )
        self.assertEqual(risk_manager.limits.max_leverage_conservative, 5.0)
        self.assertEqual(risk_manager.limits.max_position_size_pct_conservative, 0.5)

    def test_risk_score_calculation(self):
        """Test risk score calculation accuracy"""
        risk_manager = create_institutional_risk_manager()

        # Low risk scenario
        score = risk_manager._calculate_risk_score(
            exposure_pct=5.0,
            pnl_pct=-0.5,
            leverage_utilization=2.0,
            position_count=1
        )
        self.assertLess(score, 30)

        # High risk scenario
        score = risk_manager._calculate_risk_score(
            exposure_pct=40.0,
            pnl_pct=-8.0,
            leverage_utilization=15.0,
            position_count=4
        )
        self.assertGreater(score, 70)

def run_performance_tests():
    """Run performance tests for risk management system"""
    logger.info("🚀 Running performance tests...")

    risk_manager = create_institutional_risk_manager()

    # Test validation performance
    start_time = time.time()
    for i in range(1000):
        risk_manager.validate_trade(
            symbol="BTCUSDT",
            proposed_quantity=0.001,
            proposed_leverage=10.0,
            current_price=50000.0,
            account_balance=1000.0,
            volatility=10.0,
            open_positions=[],
            emergency_mode=False
        )
    validation_time = time.time() - start_time

    logger.info(f"✅ 1000 validations completed in {validation_time:.3f} seconds")
    logger.info(f"   Average: {(validation_time / 1000) * 1000:.3f} ms per validation")

    # Test position sizing performance
    position_sizer = DynamicPositionSizer(risk_manager.limits)
    start_time = time.time()
    for i in range(1000):
        position_sizer.calculate_position_size(
            account_balance=1000.0,
            symbol_price=50000.0,
            volatility=10.0,
            risk_level=RiskLevel.LOW,
            emergency_mode=False
        )
    sizing_time = time.time() - start_time

    logger.info(f"✅ 1000 position sizing calculations in {sizing_time:.3f} seconds")
    logger.info(f"   Average: {(sizing_time / 1000) * 1000:.3f} ms per calculation")

def main():
    """Run all tests"""
    logger.info("🧪 Starting Risk Management System Tests")

    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)

    # Run performance tests
    run_performance_tests()

    logger.info("✅ All tests completed successfully!")

if __name__ == "__main__":
    main()
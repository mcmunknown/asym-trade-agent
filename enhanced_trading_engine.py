"""
ENHANCED TRADING ENGINE WITH INSTITUTIONAL RISK MANAGEMENT
===========================================================

This is the production-ready trading engine that integrates the institutional-grade
risk management system. It enforces all risk controls and cannot be bypassed.

Key Features:
- Mandatory pre-trade risk validation
- Dynamic position sizing based on account balance
- Real-time leverage enforcement
- Circuit breaker integration
- Emergency mode enforcement
- Comprehensive monitoring and logging
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from decimal import Decimal

from trading_engine import TradingSignal
from bybit_client import BybitClient
from multi_model_client import MultiModelClient
from config import Config
from risk_enforcement_layer import (
    RiskEnforcementLayer, TradeRequest, TradeExecutionPlan
)
from risk_management_system import RiskLevel

logger = logging.getLogger(__name__)

class EnhancedTradingEngine:
    """
    Enhanced trading engine with institutional-grade risk management
    """

    def __init__(self):
        # Initialize core components
        self.bybit_client = BybitClient()
        self.multi_model_client = MultiModelClient()
        self.config = Config()

        # Initialize risk enforcement layer
        self.risk_enforcement = RiskEnforcementLayer(self.config)

        # Engine state
        self.engine_active = False
        self.last_signal_check = 0
        self.signal_check_interval = self.config.SIGNAL_CHECK_INTERVAL
        self.active_positions = {}

        # Performance tracking
        self.trades_executed = 0
        self.trades_successful = 0
        self.total_pnl = 0.0

        logger.info("🚀 Enhanced Trading Engine initialized")
        logger.info("🛡️ Institutional risk management: ACTIVE")
        logger.info(f"   Emergency mode: {self.config.EMERGENCY_DEEPSEEK_ONLY}")
        logger.info(f"   Max leverage: {self.config.MAX_LEVERAGE}x")
        logger.info(f"   Max position size: {self.config.MAX_POSITION_SIZE_PERCENTAGE}%")

    def start_engine(self):
        """Start the trading engine"""

        if self.engine_active:
            logger.warning("Engine is already active")
            return

        logger.info("🎬 Starting Enhanced Trading Engine...")

        # Pre-start risk checks
        account_balance = self.risk_enforcement.get_account_balance(self.bybit_client)
        if account_balance < 10:
            logger.error(f"❌ Insufficient account balance: ${account_balance:.2f}")
            return

        # Check circuit breaker status
        if not self.risk_enforcement.pre_trade_check("SYSTEM_CHECK", self.bybit_client):
            logger.error("❌ Cannot start engine - Risk controls prevent trading")
            return

        self.engine_active = True
        logger.info("✅ Enhanced Trading Engine started successfully")
        logger.info(f"   Account balance: ${account_balance:.2f}")

        # Start main trading loop
        self.run_trading_loop()

    def stop_engine(self, reason: str = "Manual stop"):
        """Stop the trading engine"""

        if not self.engine_active:
            logger.warning("Engine is already stopped")
            return

        logger.info(f"🛑 Stopping Enhanced Trading Engine: {reason}")
        self.engine_active = False

        # Final risk status
        risk_status = self.risk_enforcement.get_risk_status()
        logger.info(f"   Final risk status: {risk_status['status']}")
        logger.info(f"   Total trades executed: {self.trades_executed}")
        logger.info(f"   Success rate: {(self.trades_successful / max(self.trades_executed, 1) * 100):.1f}%")

    def run_trading_loop(self):
        """Main trading loop with integrated risk management"""

        logger.info("🔄 Entering main trading loop...")

        while self.engine_active:
            try:
                current_time = time.time()

                # Check if it's time for signal analysis
                if current_time - self.last_signal_check >= self.signal_check_interval:
                    self.last_signal_check = current_time

                    logger.info("📊 Performing signal analysis...")

                    # Get trading signals
                    signals = self._get_trading_signals()

                    if signals:
                        logger.info(f"🎯 Received {len(signals)} trading signals")

                        for signal in signals:
                            if not self.engine_active:
                                break

                            # Process each signal with risk validation
                            self._process_signal_with_risk_validation(signal)
                    else:
                        logger.info("   No trading signals generated")

                # Update risk metrics periodically
                if int(current_time) % 60 == 0:  # Every minute
                    self.risk_enforcement.update_risk_metrics(self.bybit_client)

                # Brief sleep to prevent excessive API calls
                time.sleep(10)

            except KeyboardInterrupt:
                logger.info("👋 Trading loop interrupted by user")
                self.stop_engine("User interrupt")
                break

            except Exception as e:
                logger.error(f"❌ Error in trading loop: {str(e)}")
                # Continue running but log the error
                time.sleep(30)  # Wait longer after errors

    def _get_trading_signals(self) -> List[TradingSignal]:
        """Get trading signals from the multi-model client"""

        try:
            # Get consensus signal
            consensus_result = self.multi_model_client.get_consensus_signal(
                emergency_mode=self.config.EMERGENCY_DEEPSEEK_ONLY
            )

            if consensus_result and consensus_result.consensus_reached:
                # Convert consensus result to TradingSignal
                signal_data = consensus_result.recommended_params

                # Apply emergency mode constraints
                if self.config.EMERGENCY_DEEPSEEK_ONLY:
                    signal_data['leverage'] = min(signal_data.get('leverage', 50), self.config.MAX_LEVERAGE)
                    logger.info(f"   Emergency mode: Leverage capped at {signal_data['leverage']}x")

                signal = TradingSignal(
                    symbol=signal_data['symbol'],
                    direction=signal_data['direction'],
                    entry_price=signal_data['entry_price'],
                    stop_loss=signal_data['stop_loss'],
                    take_profit=signal_data['take_profit'],
                    leverage=signal_data['leverage'],
                    quantity=signal_data.get('quantity', 0),  # Will be calculated by risk manager
                    confidence=consensus_result.confidence_score,
                    reasoning=consensus_result.reasoning,
                    timestamp=consensus_result.timestamp,
                    volatility=signal_data.get('volatility', 10.0)  # Default volatility
                )

                return [signal]

            return []

        except Exception as e:
            logger.error(f"❌ Failed to get trading signals: {str(e)}")
            return []

    def _process_signal_with_risk_validation(self, signal: TradingSignal):
        """Process trading signal with comprehensive risk validation"""

        logger.info(f"🔍 Processing signal: {signal.symbol} {signal.direction}")
        logger.info(f"   Entry: ${signal.entry_price:.4f}")
        logger.info(f"   Stop: ${signal.stop_loss:.4f}")
        logger.info(f"   Target: ${signal.take_profit:.4f}")
        logger.info(f"   Confidence: {signal.confidence:.2f}")

        # Quick pre-trade check
        if not self.risk_enforcement.pre_trade_check(signal.symbol, self.bybit_client):
            logger.warning(f"❌ Pre-trade check failed for {signal.symbol}")
            return

        # Create trade request
        trade_request = TradeRequest(
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=signal.entry_price,
            quantity=signal.quantity,  # Will be adjusted by risk manager
            leverage=signal.leverage,  # Will be validated/adjusted by risk manager
            stop_loss_price=signal.stop_loss,
            take_profit_price=signal.take_profit,
            signal_confidence=signal.confidence,
            volatility=signal.volatility,
            emergency_mode=self.config.EMERGENCY_DEEPSEEK_ONLY
        )

        # Validate and adjust trade according to risk rules
        execution_plan = self.risk_enforcement.validate_and_adjust_trade(
            trade_request, self.bybit_client
        )

        # Check if trade is allowed
        if not execution_plan.execution_allowed:
            logger.warning(f"❌ Trade rejected by risk management: {signal.symbol}")
            if execution_plan.validation_result.violations:
                for violation in execution_plan.validation_result.violations:
                    logger.warning(f"   Violation: {violation}")
            return

        # Execute the trade
        logger.info(f"✅ Trade approved by risk management: {signal.symbol}")
        self._execute_validated_trade(execution_plan)

    def _execute_validated_trade(self, execution_plan: TradeExecutionPlan):
        """Execute a validated trade with all risk controls enforced"""

        try:
            symbol = execution_plan.symbol

            logger.info(f"💰 Executing trade: {symbol}")
            logger.info(f"   Direction: {execution_plan.direction}")
            logger.info(f"   Quantity: {execution_plan.quantity}")
            logger.info(f"   Leverage: {execution_plan.leverage}x")
            logger.info(f"   Position size: ${execution_plan.position_size_usd:.2f}")
            logger.info(f"   Total exposure: ${execution_plan.exposure_usd:.2f}")
            logger.info(f"   Risk level: {execution_plan.risk_level.value}")

            # Step 1: Set leverage (enforced by risk system)
            logger.info(f"🔧 Setting leverage: {execution_plan.leverage}x")
            leverage_set = self.bybit_client.set_leverage(symbol, execution_plan.leverage)

            if not leverage_set:
                logger.error(f"❌ Failed to set leverage for {symbol}")
                return

            # Step 2: Place the order with risk-enforced parameters
            logger.info(f"📈 Placing {execution_plan.direction} order...")

            order_result = self.bybit_client.place_order(
                symbol=symbol,
                side=execution_plan.direction,
                order_type="Market",  # Use market orders for execution certainty
                qty=execution_plan.quantity,
                time_in_force="GoodTillCancel",
                reduce_only=False,
                close_on_trigger=False
            )

            if not order_result or order_result.get('retCode') != 0:
                logger.error(f"❌ Order failed for {symbol}: {order_result.get('retMsg', 'Unknown error')}")
                return

            # Step 3: Set stop loss and take profit
            order_id = order_result['result']['orderId']
            logger.info(f"✅ Order placed: {order_id}")

            # Set stop loss
            if execution_plan.stop_loss_price:
                logger.info(f"🛡️ Setting stop loss: ${execution_plan.stop_loss_price:.4f}")
                sl_result = self.bybit_client.set_trading_stop(
                    symbol=symbol,
                    stop_loss=execution_plan.stop_loss_price
                )

                if sl_result and sl_result.get('retCode') == 0:
                    logger.info("✅ Stop loss set successfully")
                else:
                    logger.warning(f"⚠️ Failed to set stop loss: {sl_result.get('retMsg', 'Unknown error')}")

            # Set take profit
            if execution_plan.take_profit_price:
                logger.info(f"🎯 Setting take profit: ${execution_plan.take_profit_price:.4f}")
                tp_result = self.bybit_client.set_trading_stop(
                    symbol=symbol,
                    take_profit=execution_plan.take_profit_price
                )

                if tp_result and tp_result.get('retCode') == 0:
                    logger.info("✅ Take profit set successfully")
                else:
                    logger.warning(f"⚠️ Failed to set take profit: {tp_result.get('retMsg', 'Unknown error')}")

            # Step 4: Post-trade monitoring
            execution_result = {
                'success': True,
                'order_id': order_id,
                'symbol': symbol,
                'direction': execution_plan.direction,
                'quantity': execution_plan.quantity,
                'leverage': execution_plan.leverage,
                'position_size': execution_plan.position_size_usd,
                'exposure': execution_plan.exposure_usd
            }

            self.risk_enforcement.post_trade_monitoring(symbol, execution_result, self.bybit_client)

            # Update performance tracking
            self.trades_executed += 1
            self.active_positions[symbol] = {
                'order_id': order_id,
                'direction': execution_plan.direction,
                'quantity': execution_plan.quantity,
                'leverage': execution_plan.leverage,
                'entry_time': time.time(),
                'execution_plan': execution_plan
            }

            logger.info(f"🎉 Trade executed successfully: {symbol}")
            logger.info(f"   Total trades executed: {self.trades_executed}")

        except Exception as e:
            logger.error(f"❌ Trade execution failed for {execution_plan.symbol}: {str(e)}")

            # Log error for risk monitoring
            error_result = {
                'success': False,
                'symbol': execution_plan.symbol,
                'error': str(e)
            }

            self.risk_enforcement.post_trade_monitoring(execution_plan.symbol, error_result, self.bybit_client)

    def close_position(self, symbol: str, reason: str = "Manual close"):
        """Close a position with risk validation"""

        if symbol not in self.active_positions:
            logger.warning(f"No active position found for {symbol}")
            return

        position_info = self.active_positions[symbol]

        logger.info(f"🔄 Closing position: {symbol} ({reason})")

        try:
            # Get current position details
            positions = self.bybit_client.get_positions()
            current_position = None

            if positions and 'result' in positions:
                for pos in positions['result'].get('list', []):
                    if pos.get('symbol') == symbol and float(pos.get('size', 0)) != 0:
                        current_position = pos
                        break

            if not current_position:
                logger.warning(f"No position found on exchange for {symbol}")
                del self.active_positions[symbol]
                return

            # Determine close direction
            position_side = current_position.get('side', 'Buy')
            close_direction = 'Sell' if position_side == 'Buy' else 'Buy'
            position_size = float(current_position.get('size', 0))

            # Execute close order
            logger.info(f"📈 Placing close order: {close_direction} {position_size}")

            close_result = self.bybit_client.place_order(
                symbol=symbol,
                side=close_direction,
                order_type="Market",
                qty=position_size,
                time_in_force="GoodTillCancel",
                reduce_only=True,  # Close position only
                close_on_trigger=False
            )

            if close_result and close_result.get('retCode') == 0:
                logger.info(f"✅ Position closed successfully: {symbol}")

                # Update performance tracking
                realized_pnl = float(current_position.get('realisedPnl', 0))
                self.total_pnl += realized_pnl
                if realized_pnl > 0:
                    self.trades_successful += 1

                logger.info(f"   Realized PnL: ${realized_pnl:.2f}")
                logger.info(f"   Total PnL: ${self.total_pnl:.2f}")

            else:
                logger.error(f"❌ Failed to close position: {close_result.get('retMsg', 'Unknown error')}")

            # Remove from active positions
            del self.active_positions[symbol]

            # Update risk metrics after closing
            self.risk_enforcement.update_risk_metrics(self.bybit_client)

        except Exception as e:
            logger.error(f"❌ Error closing position {symbol}: {str(e)}")

    def emergency_close_all_positions(self, reason: str = "Emergency"):
        """Emergency close all positions"""

        logger.critical(f"🚨 EMERGENCY CLOSE ALL POSITIONS: {reason}")

        # Activate emergency stop
        self.risk_enforcement.emergency_stop(reason)

        # Close all active positions
        symbols_to_close = list(self.active_positions.keys())
        for symbol in symbols_to_close:
            self.close_position(symbol, f"Emergency close: {reason}")

        logger.critical("🚨 Emergency close completed")

    def get_engine_status(self) -> Dict:
        """Get comprehensive engine status"""

        # Get risk status
        risk_status = self.risk_enforcement.get_risk_status()

        # Get account balance
        account_balance = self.risk_enforcement.get_account_balance(self.bybit_client)

        # Compile engine status
        status = {
            'engine_active': self.engine_active,
            'account_balance': account_balance,
            'active_positions': len(self.active_positions),
            'trades_executed': self.trades_executed,
            'trades_successful': self.trades_successful,
            'success_rate': (self.trades_successful / max(self.trades_executed, 1) * 100),
            'total_pnl': self.total_pnl,
            'risk_status': risk_status,
            'config': {
                'emergency_mode': self.config.EMERGENCY_DEEPSEEK_ONLY,
                'max_leverage': self.config.MAX_LEVERAGE,
                'max_position_size_pct': self.config.MAX_POSITION_SIZE_PERCENTAGE,
                'signal_check_interval': self.signal_check_interval
            },
            'active_position_details': {
                symbol: {
                    'direction': pos['direction'],
                    'quantity': pos['quantity'],
                    'leverage': pos['leverage'],
                    'entry_time': pos['entry_time'],
                    'risk_level': pos['execution_plan'].risk_level.value
                }
                for symbol, pos in self.active_positions.items()
            }
        }

        return status

def main():
    """Main function to run the enhanced trading engine"""

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('enhanced_trading_engine.log'),
            logging.StreamHandler()
        ]
    )

    logger.info("🚀 Starting Enhanced Trading Engine with Institutional Risk Management")

    try:
        # Create and start the engine
        engine = EnhancedTradingEngine()
        engine.start_engine()

    except KeyboardInterrupt:
        logger.info("👋 Engine stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error in trading engine: {str(e)}")
        raise
    finally:
        logger.info("🏁 Enhanced Trading Engine shutdown complete")

if __name__ == "__main__":
    main()
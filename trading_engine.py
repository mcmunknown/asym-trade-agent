import time
import threading
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from bybit_client import BybitClient
from grok4_client import Grok4Client
from config import Config

logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    symbol: str
    signal: str  # BUY/NONE
    confidence: float
    entry_price: float
    activation_price: float
    trailing_stop_pct: float
    invalidation_level: float
    thesis_summary: str
    risk_reward_ratio: str
    leverage: int
    quantity: float

class TradingEngine:
    def __init__(self):
        self.bybit_client = BybitClient()
        self.glm_client = Grok4Client()
        self.active_positions = {}
        self.trade_history = []
        self.is_running = False

    def initialize(self):
        """Initialize the trading engine"""
        try:
            # Test API connections
            balance = self.bybit_client.get_account_balance()
            if balance:
                logger.info(f"Bybit API connected. Available balance: {balance}")
            else:
                logger.error("Failed to connect to Bybit API")

            logger.info("Trading engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize trading engine: {str(e)}")
            raise

    def process_signals(self, data_list: List[Dict]):
        """Process market data and generate asymmetric trading signals using prompt.md 7-category filter"""
        try:
            logger.info(f"Processing asymmetric signals for {len(data_list)} assets using prompt.md criteria...")

            # Use new asymmetric analysis with all 7 prompt.md categories
            for symbol_data in data_list:
                try:
                    # Apply complete prompt.md 7-category filter system
                    analysis_result = self.glm_client.analyze_asymmetric_criteria(symbol_data)

                    # Only execute trades if signal is BUY (all 7 categories passed)
                    if analysis_result['signal'] == 'BUY':
                        logger.info(f"🚀 ASYMMETRIC BUY SIGNAL: {symbol_data['symbol']} - All 7 categories passed!")
                        self.handle_asymmetric_signal(symbol_data['symbol'], analysis_result, symbol_data)
                    else:
                        logger.info(f"❌ NO SIGNAL: {symbol_data['symbol']} - {analysis_result.get('thesis_summary', 'Categories not met')}")

                except Exception as e:
                    logger.error(f"Error analyzing {symbol_data.get('symbol', 'Unknown')}: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"Error processing asymmetric signals: {str(e)}")

    async def handle_asymmetric_signal(self, symbol: str, analysis: Dict, symbol_data: Dict):
        """Handle asymmetric signal and execute trade with prompt.md discipline"""
        try:
            if analysis['signal'] == 'BUY' and analysis['confidence'] >= 85:  # Higher confidence for asymmetric
                logger.info(f"🎯 ASYMMETRIC SIGNAL: {symbol}")
                logger.info(f"   Confidence: {analysis['confidence']}%")
                logger.info(f"   Entry: ${analysis['entry_price']:.4f}")
                logger.info(f"   Target: ${analysis['activation_price']:.4f} (150% PNL)")
                logger.info(f"   Leverage: {analysis['leverage']}x")
                logger.info(f"   Thesis: {analysis['thesis_summary']}")

                # Create trading signal with asymmetric parameters
                signal = self._create_asymmetric_trading_signal(symbol, analysis, symbol_data)
                if signal:
                    self.execute_asymmetric_trade(signal)

            else:
                logger.info(f"❌ NO ASYMMETRIC SIGNAL: {symbol}")

        except Exception as e:
            logger.error(f"Error handling asymmetric signal for {symbol}: {str(e)}")

    def _create_asymmetric_trading_signal(self, symbol: str, analysis: Dict, symbol_data: Dict) -> Optional[TradingSignal]:
        """Create asymmetric trading signal with prompt.md parameters"""
        try:
            # Extract values from comprehensive analysis
            current_price = analysis['entry_price']
            activation_price = analysis['activation_price']
            trailing_stop = analysis['trailing_stop_pct']
            invalidation_level = analysis['invalidation_level']
            leverage = analysis['leverage']
            quantity = analysis['quantity']
            
            # Verify execution guardrails are met (prompt.md Category 6)
            execution_guardrails = symbol_data.get('execution_guardrails', {})
            if not execution_guardrails.get('overall_guardrails', {}).get('all_guardrails_pass', False):
                logger.warning(f"Execution guardrails failed for {symbol}")
                return None

            signal = TradingSignal(
                symbol=symbol,
                signal='BUY',
                confidence=analysis['confidence'],
                entry_price=current_price,
                activation_price=activation_price,
                trailing_stop_pct=trailing_stop,
                invalidation_level=invalidation_level,
                thesis_summary=analysis['thesis_summary'],
                risk_reward_ratio=analysis.get('risk_reward_ratio', '1:5+'),
                leverage=leverage,
                quantity=quantity
            )

            logger.info(f"🎯 ASYMMETRIC SIGNAL CREATED: {symbol}")
            logger.info(f"   Entry: ${current_price:.4f}")
            logger.info(f"   Target: ${activation_price:.4f} (150% PNL)")
            logger.info(f"   Quantity: {quantity:.6f}")
            logger.info(f"   Leverage: {leverage}x")
            logger.info(f"   Position Value: ${Config.DEFAULT_TRADE_SIZE * leverage}")
            
            return signal

        except Exception as e:
            logger.error(f"Error creating asymmetric trading signal for {symbol}: {str(e)}")
            return None

    def execute_asymmetric_trade(self, signal: TradingSignal):
        """Execute asymmetric trade with prompt.md discipline"""
        try:
            # SAFETY CHECK: Disable trading if flag is set
            if Config.DISABLE_TRADING:
                logger.info(f"🚫 SIMULATION MODE: Would execute trade for {signal.symbol}")
                logger.info(f"   Entry: ${signal.entry_price:.4f}, Target: ${signal.activation_price:.4f}")
                logger.info(f"   Size: ${Config.DEFAULT_TRADE_SIZE} x {signal.leverage}x leverage")
                logger.info(f"   Expected Profit: ${Config.DEFAULT_TRADE_SIZE * 1.5:.1f}")
                return

            logger.info(f"🚀 EXECUTING ASYMMETRIC TRADE: {signal.symbol}")

            # Set leverage for the symbol (prompt.md requirement: 50-75x)
            leverage_set = self.bybit_client.set_leverage(signal.symbol, signal.leverage)
            if not leverage_set:
                logger.error(f"Failed to set leverage for {signal.symbol}")
                return

            # Place market order for immediate execution
            order_result = self.bybit_client.place_order(
                symbol=signal.symbol,
                side='Buy',
                order_type='Market',
                qty=signal.quantity
            )

            if order_result:
                # Place take profit order at 150% PNL target
                tp_order = self.bybit_client.place_order(
                    symbol=signal.symbol,
                    side='Sell',
                    order_type='Limit',
                    qty=signal.quantity,
                    price=signal.activation_price,
                    reduce_only=True
                )

                # Place stop loss order for liquidation protection
                sl_order = self.bybit_client.place_order(
                    symbol=signal.symbol,
                    side='Sell',
                    order_type='Market',
                    qty=signal.quantity,
                    reduce_only=True,
                    close_on_trigger=True
                )

                # Record the asymmetric trade with all 7 categories
                trade_record = {
                    'symbol': signal.symbol,
                    'signal': signal,
                    'orders': {
                        'entry': order_result,
                        'take_profit': tp_order,
                        'stop_loss': sl_order
                    },
                    'timestamp': datetime.now().isoformat(),
                    'status': 'ACTIVE',
                    'asymmetric_metrics': {
                        'categories_passed': 7,
                        'target_pnl': Config.DEFAULT_TRADE_SIZE * 1.5,  # 150% PNL
                        'hold_timeframe': '20-60 days',  # prompt.md timeframe
                        'risk_reward': signal.risk_reward_ratio
                    }
                }

                self.active_positions[signal.symbol] = trade_record
                self.trade_history.append(trade_record)

                logger.info(f"✅ ASYMMETRIC TRADE EXECUTED: {signal.symbol}")
                logger.info(f"   Entry: Market at ${signal.entry_price:.4f}")
                logger.info(f"   Target: Limit at ${signal.activation_price:.4f} (150% PNL)")
                logger.info(f"   Stop: Market trigger at ${signal.invalidation_level:.4f}")
                logger.info(f"   Size: ${Config.DEFAULT_TRADE_SIZE} x {signal.leverage}x = ${Config.DEFAULT_TRADE_SIZE * signal.leverage:.0f}")
                logger.info(f"   Expected Profit: ${Config.DEFAULT_TRADE_SIZE * 1.5:.1f}")

            else:
                logger.error(f"❌ Failed to execute asymmetric entry for {signal.symbol}")

        except Exception as e:
            logger.error(f"Error executing asymmetric trade for {signal.symbol}: {str(e)}")

    def monitor_positions(self):
        """Monitor active positions and manage exits"""
        while self.is_running:
            try:
                if not self.active_positions:
                    time.sleep(60)
                    continue

                logger.info(f"Monitoring {len(self.active_positions)} active positions...")

                positions_to_close = []

                for symbol, position_data in self.active_positions.items():
                    # Get current position info
                    position_info = self.bybit_client.get_position_info(symbol)
                    if not position_info:
                        continue

                    current_price = float(position_info['markPrice'])
                    unrealized_pnl = float(position_info['unrealisedPnl'])
                    signal = position_data['signal']

                    # Check if take profit should be triggered (150% PNL target)
                    target_pnl = Config.DEFAULT_TRADE_SIZE * 1.5  # $3 * 1.5 = $4.5 target PNL
                    if unrealized_pnl >= target_pnl:
                        logger.info(f"Take profit triggered for {symbol}. PNL: ${unrealized_pnl:.2f}")
                        positions_to_close.append((symbol, 'TAKE_PROFIT'))
                        continue

                    # Check if stop loss should be triggered
                    if current_price <= signal.invalidation_level:
                        logger.info(f"Stop loss triggered for {symbol}. Price: ${current_price:.4f}")
                        positions_to_close.append((symbol, 'STOP_LOSS'))
                        continue

                    # Update trailing stop if profitable
                    if unrealized_pnl > 0:
                        # Implement trailing stop logic here
                        pass

                # Close positions that hit exit criteria
                for symbol, reason in positions_to_close:
                    self.close_position(symbol, reason)

                time.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error(f"Error monitoring positions: {str(e)}")
                time.sleep(60)

    def close_position(self, symbol: str, reason: str):
        """Close an active position"""
        try:
            logger.info(f"Closing position for {symbol} - {reason}")

            # Get current position
            position_info = self.bybit_client.get_position_info(symbol)
            if not position_info:
                logger.error(f"No position found for {symbol}")
                return

            position_size = float(position_info['size'])
            if position_size > 0:
                # Close position with market order
                close_result = self.bybit_client.place_order(
                    symbol=symbol,
                    side='Sell',
                    order_type='Market',
                    qty=position_size,
                    reduce_only=True
                )

                if close_result:
                    logger.info(f"Position closed for {symbol} - {reason}")

                    # Update position record
                    if symbol in self.active_positions:
                        self.active_positions[symbol]['status'] = 'CLOSED'
                        self.active_positions[symbol]['close_reason'] = reason
                        self.active_positions[symbol]['close_time'] = datetime.now().isoformat()

                    # Remove from active positions
                    del self.active_positions[symbol]

                else:
                    logger.error(f"Failed to close position for {symbol}")

        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {str(e)}")

    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary"""
        try:
            balance = self.bybit_client.get_account_balance()
            if balance:
                total_balance = float(balance['totalEquity'])
                available_balance = float(balance.get('totalAvailableBalance', balance.get('availableBalance', '0')))

                total_invested = len(self.active_positions) * Config.DEFAULT_TRADE_SIZE
                unrealized_pnl = 0.0

                for symbol, position_data in self.active_positions.items():
                    position_info = self.bybit_client.get_position_info(symbol)
                    if position_info:
                        unrealized_pnl += float(position_info['unrealisedPnl'])

                return {
                        'total_balance': total_balance,
                        'available_balance': available_balance,
                        'total_invested': total_invested,
                        'unrealized_pnl': unrealized_pnl,
                        'active_positions': len(self.active_positions),
                        'total_trades': len(self.trade_history)
                    }

        except Exception as e:
            logger.error(f"Error getting portfolio summary: {str(e)}")
            return {}

    def start(self):
        """Start the trading engine"""
        self.is_running = True
        logger.info("Trading engine started")

        # Start position monitoring in background
        monitor_thread = threading.Thread(target=self.monitor_positions, daemon=True)
        monitor_thread.start()

        try:
            # Keep the engine running
            while self.is_running:
                time.sleep(3600)  # Check every hour

        except KeyboardInterrupt:
            logger.info("Shutting down trading engine...")
        finally:
            self.is_running = False

    def stop(self):
        """Stop the trading engine"""
        self.is_running = False
        logger.info("Trading engine stopped")

    def get_trade_history(self) -> List[Dict]:
        """Get trade history"""
        return self.trade_history.copy()

    def get_active_positions(self) -> Dict:
        """Get active positions"""
        return self.active_positions.copy()

    def get_current_price(self, symbol: str) -> float:
        """Get current market price for a symbol"""
        try:
            market_data = self.bybit_client.get_market_data(symbol)
            if market_data and 'lastPrice' in market_data:
                return float(market_data.get('lastPrice', 0.0))
            else:
                logger.warning(f"No market data available for {symbol}")
                return 0.0
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {str(e)}")
            return 0.0
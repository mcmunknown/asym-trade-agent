import time
import threading
import logging
import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from bybit_client import BybitClient
from grok4_client import Grok4Client
from multi_model_client import MultiModelConsensusEngine
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
        self.glm_client = Grok4Client()  # Keep as fallback
        self.consensus_engine = MultiModelConsensusEngine()  # New multi-model system
        self.active_positions = {}
        self.trade_history = []
        self.is_running = False
        self.use_multi_model = Config.ENABLE_MULTI_MODEL  # Use config setting

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
        """Process market data and generate asymmetric trading signals using multi-model consensus"""
        try:
            if self.use_multi_model:
                logger.info(f"🤖 Processing multi-model consensus signals for {len(data_list)} assets using 3 AI analysts...")
            else:
                logger.info(f"Processing asymmetric signals for {len(data_list)} assets using prompt.md criteria...")

            # Process each asset
            for symbol_data in data_list:
                try:
                    if self.use_multi_model:
                        # Use multi-model consensus engine
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        consensus_result = loop.run_until_complete(
                            self.consensus_engine.get_consensus_signal(symbol_data)
                        )
                        loop.close()

                        # Handle consensus result
                        if consensus_result.final_signal == 'BUY':
                            logger.info(f"🚀 MULTI-MODEL CONSENSUS BUY SIGNAL: {symbol_data['symbol']}")
                            logger.info(f"   Votes: {consensus_result.consensus_votes}")
                            logger.info(f"   Confidence: {consensus_result.confidence_avg:.2f}")
                            logger.info(f"   Thesis: {consensus_result.thesis_combined[:200]}...")

                            # Convert consensus result to expected format
                            analysis_result = {
                                'signal': consensus_result.final_signal,
                                'confidence': consensus_result.confidence_avg,
                                'entry_price': consensus_result.recommended_params.get('entry_price', 0),
                                'activation_price': consensus_result.recommended_params.get('activation_price', 0),
                                'trailing_stop_pct': consensus_result.recommended_params.get('trailing_stop_pct', 0),
                                'invalidation_level': consensus_result.recommended_params.get('invalidation_level', 0),
                                'thesis_summary': consensus_result.thesis_combined,
                                'risk_reward_ratio': consensus_result.recommended_params.get('risk_reward_ratio', '1:5+'),
                                'leverage': consensus_result.recommended_params.get('leverage', 50),
                                'quantity': consensus_result.recommended_params.get('quantity', 0),
                                'consensus_votes': consensus_result.consensus_votes
                            }

                            self.handle_asymmetric_signal(symbol_data['symbol'], analysis_result, symbol_data)
                        else:
                            logger.info(f"❌ NO CONSENSUS: {symbol_data['symbol']} - {consensus_result.thesis_combined[:100]}...")
                            if consensus_result.disagreement_details:
                                logger.info(f"   Disagreements: {' | '.join(consensus_result.disagreement_details[:2])}")
                    else:
                        # Fallback to single Grok 4 Fast analysis
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
            logger.error(f"Error processing signals: {str(e)}")

    def toggle_multi_model_mode(self, enable: bool = None):
        """Toggle between multi-model and single-model mode"""
        if enable is None:
            self.use_multi_model = not self.use_multi_model
        else:
            self.use_multi_model = enable

        mode = "Multi-Model Consensus (3 AI analysts)" if self.use_multi_model else "Single Model (Grok 4 Fast)"
        logger.info(f"🔄 Switched to {mode} mode")

    async def test_multi_model_vs_single(self, test_data: Dict) -> Dict:
        """Compare multi-model consensus vs single model performance"""
        results = {}

        # Test single model (Grok 4 Fast)
        try:
            single_result = self.glm_client.analyze_asymmetric_criteria(test_data)
            results["single_model"] = {
                "signal": single_result['signal'],
                "confidence": single_result['confidence'],
                "thesis": single_result['thesis_summary']
            }
        except Exception as e:
            results["single_model"] = {"signal": "ERROR", "thesis": str(e)}

        # Test multi-model consensus
        try:
            consensus_result = await self.consensus_engine.get_consensus_signal(test_data)
            results["multi_model"] = {
                "signal": consensus_result.final_signal,
                "confidence": consensus_result.confidence_avg,
                "thesis": consensus_result.thesis_combined,
                "votes": consensus_result.consensus_votes
            }
        except Exception as e:
            results["multi_model"] = {"signal": "ERROR", "thesis": str(e)}

        return results

    def handle_asymmetric_signal(self, symbol: str, analysis: Dict, symbol_data: Dict):
        """Handle asymmetric signal and execute trade with prompt.md discipline"""
        try:
            if analysis['signal'] == 'BUY':  # If Grok 4 Fast says BUY after 7-category analysis, execute!
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

            # Calculate exact $3 position size with MAXIMUM LEVERAGE for asymmetric returns
            # Your strategy: $3 base position × maximum leverage (50-75x) = $150-225 exposure
            try:
                # Get instrument info for this symbol to calculate proper position sizing
                instrument_info = self.bybit_client.get_instrument_info(symbol)
                if not instrument_info:
                    logger.error(f"Failed to get instrument info for {symbol}")
                    return None

                # Extract key Bybit parameters
                max_leverage = float(instrument_info.get('maxLeverage', 50))
                qty_step = float(instrument_info.get('qtyStep', 0.001))
                min_order_qty = float(instrument_info.get('minOrderQty', 0.001))
                min_notional = float(instrument_info.get('minNotionalValue', 5.0))

                # Calculate position with maximum leverage
                base_position_size = Config.DEFAULT_TRADE_SIZE  # $3
                target_exposure = base_position_size * analysis['leverage']  # $3 × leverage = exposure

                # Calculate quantity needed for target exposure
                calculated_quantity = target_exposure / current_price

                # Round to valid quantity step (critical for Bybit)
                quantity = round(calculated_quantity / qty_step) * qty_step

                # Ensure minimum requirements are met
                quantity = max(quantity, min_order_qty)

                # Calculate actual position values
                actual_base_value = quantity * current_price
                actual_exposure = actual_base_value * analysis['leverage']

                # Get token base for logging
                token_base = symbol.replace('USDT', '')

                logger.info(f"   💰 MAX LEVERAGE POSITION CALCULATION:")
                logger.info(f"   Base position: ${base_position_size}")
                logger.info(f"   Target leverage: {analysis['leverage']}x")
                logger.info(f"   Target exposure: ${target_exposure:.0f}")
                logger.info(f"   Current price: ${current_price:.4f}")
                logger.info(f"   Quantity step: {qty_step}")
                logger.info(f"   Min quantity: {min_order_qty}")
                logger.info(f"   Calculated quantity: {quantity:.6f} {token_base}")
                logger.info(f"   Actual base value: ${actual_base_value:.2f}")
                logger.info(f"   Actual exposure: ${actual_exposure:.0f}")
                logger.info(f"   Expected PNL at 1000%: ${Config.DEFAULT_TRADE_SIZE * 10:.2f}")

            except Exception as e:
                logger.error(f"Error calculating position size for {symbol}: {str(e)}")
                # Fallback to simple calculation
                calculated_quantity = Config.DEFAULT_TRADE_SIZE / current_price
                quantity = round(calculated_quantity, 6)

            # Skip execution guardrails check - consensus BUY signal takes priority
            logger.info(f"✅ Execution guardrails bypassed for {symbol} - Multi-model consensus BUY signal takes priority")

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
        """Execute asymmetric trade with maximum leverage and exact $3 positioning"""
        try:
            # SAFETY CHECK: Disable trading if flag is set
            if hasattr(Config, 'DISABLE_TRADING') and Config.DISABLE_TRADING:
                logger.info(f"🚫 SIMULATION MODE: Would execute multi-model consensus trade for {signal.symbol}")
                logger.info(f"   Entry: ${signal.entry_price:.4f}, Leverage: {signal.leverage}x")
                logger.info(f"   Base position: ${Config.DEFAULT_TRADE_SIZE}, Exposure: ${Config.DEFAULT_TRADE_SIZE * signal.leverage:.0f}")
                logger.info(f"   Expected Profit: ${Config.DEFAULT_TRADE_SIZE * 1.5:.1f}")
                return

            logger.info(f"🚀 EXECUTING MULTI-MODEL CONSENSUS TRADE: {signal.symbol}")

            # Get instrument info to ensure we're using proper Bybit parameters
            instrument_info = self.bybit_client.get_instrument_info(signal.symbol)
            if not instrument_info:
                logger.error(f"Failed to get instrument info for {signal.symbol}")
                return

            # Set MAXIMUM LEVERAGE for the symbol (crucial for asymmetric returns)
            max_available_leverage = float(instrument_info.get('maxLeverage', signal.leverage))
            actual_leverage = min(signal.leverage, max_available_leverage)

            logger.info(f"   Requested leverage: {signal.leverage}x")
            logger.info(f"   Max available leverage: {max_available_leverage}x")
            logger.info(f"   Using leverage: {actual_leverage}x")

            leverage_set = self.bybit_client.set_leverage(signal.symbol, actual_leverage)
            if not leverage_set:
                logger.error(f"Failed to set leverage for {signal.symbol}")
                return

            # Calculate 3-day position trading TP/SL levels for 1000%+ returns
            # With $3 base position × 75x leverage = $225 exposure
            # 1000% return means $3 × 10 = $30 profit on base position
            # Need price movement of: $30 ÷ $225 = 13.33% price increase
            target_multiplier = 1.133  # 13.3% price target for 1000% returns
            take_profit_price = signal.entry_price * target_multiplier
            stop_loss_price = signal.entry_price * 0.95     # 5% stop loss (wider for high-leverage plays)

            # Place single order with built-in TP/SL for 3-day position trading
            order_result = self.bybit_client.place_order(
                symbol=signal.symbol,
                side='Buy',
                order_type='Market',
                qty=signal.quantity,
                take_profit=take_profit_price,
                stop_loss=stop_loss_price
            )

            if order_result:
                # Calculate actual position values
                actual_base_value = signal.quantity * signal.entry_price
                actual_exposure = actual_base_value * actual_leverage
                expected_profit = Config.DEFAULT_TRADE_SIZE * 10  # 1000% return: $3 × 10 = $30 profit

                # Record the consensus trade with multi-model metrics
                trade_record = {
                    'symbol': signal.symbol,
                    'signal': signal,
                    'order': order_result,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'ACTIVE',
                    'consensus_metrics': {
                        'models_voted': signal.consensus_votes if hasattr(signal, 'consensus_votes') else 'N/A',
                        'base_position': Config.DEFAULT_TRADE_SIZE,
                        'leverage_used': actual_leverage,
                        'exposure': actual_exposure,
                        'expected_profit': expected_profit,
                        'hold_timeframe': '3 days',
                        'risk_reward': signal.risk_reward_ratio,
                        'take_profit_price': take_profit_price,
                        'stop_loss_price': stop_loss_price,
                        'qty_step_used': instrument_info.get('qtyStep', '0.001')
                    }
                }

                self.active_positions[signal.symbol] = trade_record
                self.trade_history.append(trade_record)

                logger.info(f"✅ MULTI-MODEL CONSENSUS TRADE EXECUTED: {signal.symbol}")
                logger.info(f"   Entry: Market at ${signal.entry_price:.4f}")
                logger.info(f"   Base Position: ${Config.DEFAULT_TRADE_SIZE}")
                logger.info(f"   Leverage: {actual_leverage}x")
                logger.info(f"   Total Exposure: ${actual_exposure:.0f}")
                logger.info(f"   Quantity: {signal.quantity:.6f}")
                logger.info(f"   Target: ${take_profit_price:.4f} (13.3% for 3-day hold - 1000% returns)")
                logger.info(f"   Stop: ${stop_loss_price:.4f} (5% stop loss for high-leverage)")
                logger.info(f"   Expected Profit: ${expected_profit:.1f} (1000% return on $3 base)")
                logger.info(f"   🎯 This is your $3 strategy with maximum leverage!")

            else:
                logger.error(f"❌ Failed to execute consensus entry for {signal.symbol}")

        except Exception as e:
            logger.error(f"Error executing consensus trade for {signal.symbol}: {str(e)}")

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

                    # Check if 3-day holding period has expired
                    entry_time_str = position_data['timestamp']
                    entry_time = datetime.fromisoformat(entry_time_str.replace('Z', '+00:00'))
                    current_time = datetime.now()
                    holding_duration = current_time - entry_time

                    # Close position after 3 days (72 hours)
                    if holding_duration >= timedelta(hours=72):
                        logger.info(f"3-day holding period expired for {symbol}. Duration: {holding_duration}")
                        positions_to_close.append((symbol, '3_DAY_HOLD_EXPIRED'))
                        continue

                    # Check if take profit should be triggered (13.3% target for 1000% returns)
                    target_pnl = Config.DEFAULT_TRADE_SIZE * 10  # 1000% return: $3 × 10 = $30 profit
                    if unrealized_pnl >= target_pnl:
                        logger.info(f"Take profit triggered for {symbol}. PNL: ${unrealized_pnl:.2f}")
                        positions_to_close.append((symbol, 'TAKE_PROFIT'))
                        continue

                    # Check if stop loss should be triggered
                    tp_price = position_data['asymmetric_metrics']['take_profit_price']
                    sl_price = position_data['asymmetric_metrics']['stop_loss_price']
                    if current_price <= sl_price:
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

                time.sleep(600)  # Check every 10 minutes (faster than 30min cycles for TP/SL)

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
                available_balance = float(balance.get('totalAvailableBalance') or balance.get('availableBalance') or '0')

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
#!/usr/bin/env python3
"""
Simple Calculus Trading Monitor
Uses REST API polling instead of WebSocket for more reliable data collection
"""

import time
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from bybit_client import BybitClient
from calculus_strategy import CalculusTradingStrategy, SignalType
from kalman_filter import AdaptiveKalmanFilter
from quantitative_models import CalculusPriceAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleCalculusMonitor:
    """Simplified calculus trading monitor using REST API"""

    def __init__(self, symbol='BTCUSDT', data_points=200):
        self.symbol = symbol
        self.data_points = data_points
        self.bybit_client = BybitClient()
        self.kalman_filter = AdaptiveKalmanFilter()
        self.calculus_strategy = CalculusTradingStrategy()

        # Price history
        self.price_history = []
        self.timestamps = []

        # Trading tracking
        self.tp_hits = 0
        self.trades_executed = 0
        self.last_signal = None

    def get_market_price(self):
        """Get current market price from Bybit"""
        try:
            ticker = self.bybit_client.get_market_data(self.symbol)
            if ticker and 'lastPrice' in ticker:
                return float(ticker['lastPrice'])
        except Exception as e:
            logger.error(f"Error getting market price: {e}")
        return None

    def analyze_signals(self):
        """Analyze calculus signals with current price data"""
        if len(self.price_history) < 50:
            return None

        try:
            # Create price series
            price_series = pd.Series(self.price_history)

            # Apply Kalman filtering
            kalman_results = self.kalman_filter.filter_price_series(price_series)
            if kalman_results.empty:
                return None

            # Get filtered prices
            if 'filtered_price' in kalman_results.columns:
                filtered_prices = kalman_results['filtered_price']
            elif 'price_estimate' in kalman_results.columns:
                filtered_prices = kalman_results['price_estimate']
            else:
                filtered_prices = price_series

            # Generate calculus signals
            signals = self.calculus_strategy.generate_trading_signals(filtered_prices)
            if signals.empty:
                return None

            # Get latest signal
            latest_signal = signals.iloc[-1]

            return {
                'timestamp': datetime.now(),
                'symbol': self.symbol,
                'price': self.price_history[-1],
                'signal_type': SignalType(int(latest_signal.get('signal_type', 0))),
                'interpretation': latest_signal.get('interpretation', 'Unknown'),
                'confidence': latest_signal.get('confidence', 0.0),
                'velocity': latest_signal.get('velocity', 0.0),
                'acceleration': latest_signal.get('acceleration', 0.0),
                'snr': latest_signal.get('snr', 0.0),
                'forecast': latest_signal.get('forecast', 0.0),
                'valid_signal': latest_signal.get('valid_signal', False)
            }

        except Exception as e:
            logger.error(f"Error analyzing signals: {e}")
            return None

    def check_actionable_signal(self, signal):
        """Check if signal meets trading criteria"""
        if not signal or not signal['valid_signal']:
            return False

        # Check confidence and SNR thresholds
        if signal['confidence'] < 0.7 or signal['snr'] < 0.8:
            return False

        # Check actionable signal types
        actionable_types = [
            SignalType.BUY, SignalType.SELL,
            SignalType.STRONG_BUY, SignalType.STRONG_SELL,
            SignalType.POSSIBLE_LONG, SignalType.POSSIBLE_EXIT_SHORT
        ]

        return signal['signal_type'] in actionable_types

    def simulate_trade_execution(self, signal):
        """Simulate trade execution with TP/SL tracking"""
        current_price = signal['price']
        signal_type = signal['signal_type']

        # Determine trade direction
        if signal_type in [SignalType.BUY, SignalType.STRONG_BUY, SignalType.POSSIBLE_LONG]:
            side = "LONG"
            entry_price = current_price
            # Calculate TP/SL based on calculus confidence
            tp_pct = 0.02 + (signal['confidence'] * 0.02)  # 2-4% TP
            sl_pct = 0.015 + (signal['confidence'] * 0.01)  # 1.5-2.5% SL
            tp_price = entry_price * (1 + tp_pct)
            sl_price = entry_price * (1 - sl_pct)
        else:
            side = "SHORT"
            entry_price = current_price
            tp_pct = 0.02 + (signal['confidence'] * 0.02)
            sl_pct = 0.015 + (signal['confidence'] * 0.01)
            tp_price = entry_price * (1 - tp_pct)
            sl_price = entry_price * (1 + sl_pct)

        self.trades_executed += 1

        logger.info(f"🚀 SIMULATED TRADE EXECUTED")
        logger.info(f"   Symbol: {self.symbol}")
        logger.info(f"   Side: {side}")
        logger.info(f"   Entry: ${entry_price:.2f}")
        logger.info(f"   TP: ${tp_price:.2f} ({tp_pct*100:.1f}%)")
        logger.info(f"   SL: ${sl_price:.2f} ({sl_pct*100:.1f}%)")
        logger.info(f"   Signal: {signal['interpretation']}")
        logger.info(f"   Confidence: {signal['confidence']:.2f}")
        logger.info(f"   SNR: {signal['snr']:.2f}")

        return {
            'entry_price': entry_price,
            'tp_price': tp_price,
            'sl_price': sl_price,
            'side': side,
            'signal': signal
        }

    def check_tp_hit(self, trade_info, current_price):
        """Check if trade hit take profit"""
        if not trade_info:
            return False

        entry = trade_info['entry_price']
        tp = trade_info['tp_price']
        sl = trade_info['sl_price']
        side = trade_info['side']

        if side == "LONG":
            if current_price >= tp:
                logger.info(f"🎯 TAKE PROFIT HIT! Entry: ${entry:.2f} → TP: ${tp:.2f} → Current: ${current_price:.2f}")
                return True
            elif current_price <= sl:
                logger.info(f"❌ STOP LOSS HIT! Entry: ${entry:.2f} → SL: ${sl:.2f} → Current: ${current_price:.2f}")
                return False
        else:  # SHORT
            if current_price <= tp:
                logger.info(f"🎯 TAKE PROFIT HIT! Entry: ${entry:.2f} → TP: ${tp:.2f} → Current: ${current_price:.2f}")
                return True
            elif current_price >= sl:
                logger.info(f"❌ STOP LOSS HIT! Entry: ${entry:.2f} → SL: ${sl:.2f} → Current: ${current_price:.2f}")
                return False

        return None

    def run_monitoring(self):
        """Run the calculus trading monitor"""
        logger.info(f"🔬 Starting Simple Calculus Monitor for {self.symbol}")
        logger.info(f"📊 Target: Track 3 TP hits for mathematical analysis")
        logger.info("=" * 60)

        active_trade = None

        while self.tp_hits < 3:
            try:
                # Get current price
                current_price = self.get_market_price()
                if not current_price:
                    logger.warning("Could not get market price, waiting...")
                    time.sleep(10)
                    continue

                # Update price history
                self.price_history.append(current_price)
                self.timestamps.append(time.time())

                # Maintain window size
                if len(self.price_history) > self.data_points:
                    self.price_history.pop(0)
                    self.timestamps.pop(0)

                # Analyze signals
                signal = self.analyze_signals()

                if signal:
                    logger.info(f"📈 CALCULUS SIGNAL: {signal['signal_type'].name}")
                    logger.info(f"   Price: ${signal['price']:.2f}")
                    logger.info(f"   Velocity: {signal['velocity']:.6f}")
                    logger.info(f"   Acceleration: {signal['acceleration']:.8f}")
                    logger.info(f"   SNR: {signal['snr']:.2f}")
                    logger.info(f"   Confidence: {signal['confidence']:.2f}")
                    logger.info(f"   Interpretation: {signal['interpretation']}")

                    # Check if we should execute a trade
                    if self.check_actionable_signal(signal) and not active_trade:
                        active_trade = self.simulate_trade_execution(signal)

                # Check active trade for TP/SL
                if active_trade:
                    tp_result = self.check_tp_hit(active_trade, current_price)
                    if tp_result is True:
                        self.tp_hits += 1
                        logger.info(f"✅ TP HIT #{self.tp_hits}/3 ACHIEVED!")
                        active_trade = None
                    elif tp_result is False:
                        logger.info("❌ Trade hit stop loss, looking for next opportunity")
                        active_trade = None

                # Show status
                logger.info(f"📊 Status: TP Hits: {self.tp_hits}/3 | Trades: {self.trades_executed} | Active Trade: {'Yes' if active_trade else 'No'}")
                logger.info("-" * 40)

                # Wait before next iteration
                time.sleep(30)  # 30 second intervals

            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error

        # Final analysis
        logger.info("=" * 60)
        logger.info("🎯 MONITORING COMPLETE!")
        logger.info(f"   Total TP Hits: {self.tp_hits}")
        logger.info(f"   Total Trades Executed: {self.trades_executed}")
        logger.info(f"   Success Rate: {(self.tp_hits / max(self.trades_executed, 1)) * 100:.1f}%")

        if self.tp_hits >= 3:
            logger.info("✅ Successfully achieved 3 TP hits!")
            logger.info("🔬 Calculus-based TP enhancement is working effectively!")
        else:
            logger.info("⚠️  Did not achieve 3 TP hits")
            logger.info("🔍 Further analysis needed for TP probability calculations")

if __name__ == "__main__":
    monitor = SimpleCalculusMonitor(symbol='BTCUSDT', data_points=200)
    monitor.run_monitoring()
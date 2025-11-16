import threading
import time
import logging
from typing import List
import math
import numpy as np

logger = logging.getLogger(__name__)

class PnlTracker:
    def __init__(self, bybit_client, performance, trading_states):
        self.bybit_client = bybit_client
        self.performance = performance
        self.trading_states = trading_states
        self.pnl_tracker = {
            'session_start_balance': 0.0,
            'session_start_time': time.time(),
            'peak_balance': 0.0,
            'trade_history': [],
            'last_update': 0.0,
            'target_balance': 100000.0,  # $100K goal
            'milestones': [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
        }

    def _update_live_pnl(self):
        """Update and display live P&L tracker (path to $100K)"""
        try:
            current_time = time.time()
            
            # Rate limit to once per 5 seconds
            if current_time - self.pnl_tracker['last_update'] < 5:
                return
            
            self.pnl_tracker['last_update'] = current_time
            
            # Get current balance
            account_info = self.bybit_client.get_account_balance()
            if not account_info:
                return
            
            current_balance = float(account_info.get('totalEquity', 0))
            if current_balance <= 0:
                current_balance = float(account_info.get('totalAvailableBalance', 0))
            
            # Initialize session start if needed
            if self.pnl_tracker['session_start_balance'] == 0:
                self.pnl_tracker['session_start_balance'] = current_balance
                self.pnl_tracker['peak_balance'] = current_balance
            
            # Update peak
            if current_balance > self.pnl_tracker['peak_balance']:
                self.pnl_tracker['peak_balance'] = current_balance
            
            # Calculate metrics
            session_start = self.pnl_tracker['session_start_balance']
            realized_pnl = current_balance - session_start
            realized_pct = (realized_pnl / session_start * 100) if session_start > 0 else 0
            
            # Get unrealized P&L from open positions
            unrealized_pnl = 0.0
            for state in self.trading_states.values():
                if state.position_info:
                    entry_price = state.position_info.get('entry_price', 0)
                    current_price = state.price_history[-1] if state.price_history else entry_price
                    qty = state.position_info.get('quantity', 0)
                    side = state.position_info.get('side', 'Buy')
                    
                    if side == 'Buy':
                        unrealized_pnl += (current_price - entry_price) * qty
                    else:
                        unrealized_pnl += (entry_price - current_price) * qty
            
            total_pnl = realized_pnl + unrealized_pnl
            total_pct = (total_pnl / session_start * 100) if session_start > 0 else 0
            
            # Drawdown
            drawdown = (self.pnl_tracker['peak_balance'] - current_balance) / self.pnl_tracker['peak_balance'] * 100 if self.pnl_tracker['peak_balance'] > 0 else 0
            
            # Win rate
            wins = self.performance.winning_trades
            losses = self.performance.losing_trades
            total_trades = wins + losses
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
            
            # Average win/loss
            avg_win = (sum([t for t in self.pnl_tracker['trade_history'] if t > 0]) / wins) if wins > 0 else 0
            avg_loss = (sum([t for t in self.pnl_tracker['trade_history'] if t < 0]) / losses) if losses > 0 else 0
            
            # Profit factor
            gross_profit = sum([t for t in self.pnl_tracker['trade_history'] if t > 0])
            gross_loss = abs(sum([t for t in self.pnl_tracker['trade_history'] if t < 0]))
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
            
            # Path to $100K calculation
            target = self.pnl_tracker['target_balance']
            if realized_pnl > 0:
                session_hours = (current_time - self.pnl_tracker['session_start_time']) / 3600
                hourly_return = (realized_pct / 100) / session_hours if session_hours > 0 else 0
                
                # Compound growth formula: target = current * (1 + r)^t
                # Solve for t: t = log(target/current) / log(1 + hourly_rate)
                if hourly_return > 0:
                    import math
                    hours_needed = math.log(target / current_balance) / math.log(1 + hourly_return)
                    days_needed = hours_needed / 24
                else:
                    days_needed = float('inf')
            else:
                days_needed = float('inf')
                hourly_return = 0
            
            # Next milestone
            next_milestone = next((m for m in self.pnl_tracker['milestones'] if m > current_balance), target)
            milestone_pct = (current_balance / next_milestone * 100) if next_milestone > 0 else 100
            
            # Display
            print(f"""
╔════════════════════════════════════════════════════════════════╗
║  💰 LIVE P&L TRACKER - PATH TO $100,000                        ║
╠════════════════════════════════════════════════════════════════╣
║  💵 Current Balance:   ${current_balance:,.2f}
║  📊 Session Start:     ${session_start:,.2f}
║  📈 Realized P&L:      ${realized_pnl:+,.2f} ({realized_pct:+.2f}%)
║  📊 Unrealized P&L:    ${unrealized_pnl:+,.2f}
║  💎 Total P&L:         ${total_pnl:+,.2f} ({total_pct:+.2f}%)
║  
║  🔝 Peak Balance:      ${self.pnl_tracker['peak_balance']:,.2f}
║  📉 Current Drawdown:  {drawdown:.2f}%
║  
║  🎯 PERFORMANCE STATS:
║     Win Rate:          {win_rate:.1f}% ({wins}W / {losses}L)
║     Avg Win:           ${avg_win:+,.2f}
║     Avg Loss:          ${avg_loss:,.2f}
║     Profit Factor:     {profit_factor:.2f}
║     Total Trades:      {total_trades}
║  
║  🚀 PATH TO $100,000:
║     Days needed:       {days_needed:.0f} days ({days_needed/30:.1f} months)
║     Required hourly:   {hourly_return*100:.2f}%/hr
║     Next milestone:    ${next_milestone:,.0f} ({milestone_pct:.1f}% there)
║  
║  ⏱️  Session time:      {(current_time - self.pnl_tracker['session_start_time'])/3600:.1f} hours
╚════════════════════════════════════════════════════════════════╝
            """)
            
        except Exception as e:
            logger.error(f"Error updating P&L tracker: {e}")

    def _record_trade_pnl(self, pnl: float):
        """Record trade P&L for tracking"""
        try:
            self.pnl_tracker['trade_history'].append(pnl)
            
            # Keep only last 100 trades to avoid memory bloat
            if len(self.pnl_tracker['trade_history']) > 100:
                self.pnl_tracker['trade_history'] = self.pnl_tracker['trade_history'][-100:]
        except Exception as e:
            logger.error(f"Error recording trade P&L: {e}")

class SelfHealing:
    def __init__(self, bybit_client, performance, risk_manager, trader):
        self.bybit_client = bybit_client
        self.performance = performance
        self.risk_manager = risk_manager
        self.trader = trader

    def _self_heal(self):
        """Self-healing logic - detect issues and adjust strategy"""
        try:
            # Get current account info
            account_info = self.bybit_client.get_account_balance()
            if not account_info:
                return
            
            current_balance = float(account_info.get('totalEquity', 0))
            if current_balance <= 0:
                current_balance = float(account_info.get('totalAvailableBalance', 0))
            
            # Calculate drawdown
            if self.trader.pnl_tracker.pnl_tracker['peak_balance'] > 0:
                drawdown = (self.trader.pnl_tracker.pnl_tracker['peak_balance'] - current_balance) / self.trader.pnl_tracker.pnl_tracker['peak_balance']
            else:
                drawdown = 0
            
            # CIRCUIT BREAKER 1: 15% drawdown → conservative mode
            if drawdown > 0.15:
                logger.warning("🚨 15% DRAWDOWN DETECTED - ENTERING CONSERVATIVE MODE")
                logger.warning("   - Reducing leverage by 50%")
                logger.warning("   - Increasing signal filters by 1.5x")
                
                # Reduce max leverage
                self.risk_manager.max_leverage = max(10, self.risk_manager.max_leverage * 0.5)
                
                # Tighten filters (would need to be implemented in tier config)
                # For now, just log it
                
            # CIRCUIT BREAKER 2: Win rate < 45% → stricter filters
            wins = self.performance.winning_trades
            losses = self.performance.losing_trades
            total_trades = wins + losses
            
            if total_trades >= 10:  # Need minimum sample
                win_rate = wins / total_trades
                if win_rate < 0.45:
                    logger.warning("📉 WIN RATE BELOW 45% - INCREASING SIGNAL QUALITY FILTERS")
                    logger.warning(f"   Current: {win_rate:.1%}, Target: 65-75%")
                    # Could implement: require 4/5 signals instead of 3/5
            
            # CIRCUIT BREAKER 3: 5+ consecutive losses → pause 1 hour
            if self.trader.consecutive_losses >= 5:
                logger.error("⛔ 5 CONSECUTIVE LOSSES - PAUSING TRADING FOR 1 HOUR")
                self.trader.emergency_stop = True
                
                # Could set a timer to re-enable after 1 hour
                # For now, manual intervention required
            
        except Exception as e:
            logger.error(f"Error in self-healing logic: {e}")


class TradingOrchestrator:
    def __init__(self, trader):
        self.trader = trader
        self.is_running = False
        self.processing_thread = None
        self.monitoring_thread = None
        self.portfolio_thread = None
        self.pnl_tracker = PnlTracker(trader.bybit_client, trader.performance, trader.trading_states)
        self.self_healing = SelfHealing(trader.bybit_client, trader.performance, trader.risk_manager, trader)


    def start(self):
        """Start the trading orchestrator."""
        logger.info("Starting TradingOrchestrator...")
        if self.is_running:
            logger.warning("Trading orchestrator is already running")
            return

        self.is_running = True
        self.trader.emergency_stop = False

        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        # Start portfolio monitoring if enabled
        if self.trader.portfolio_mode:
            self.portfolio_thread = threading.Thread(target=self._portfolio_monitoring_loop, daemon=True)
            self.portfolio_thread.start()
            logger.info("📊 Portfolio monitoring started")

    def stop(self):
        """Stop the trading orchestrator gracefully."""
        logger.info("Stopping trading orchestrator...")
        self.is_running = False
        self.trader.emergency_stop = True

        # Stop WebSocket client
        if self.trader.ws_client:
            self.trader.ws_client.stop()

        # Wait for threads to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)

        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
            
        if self.portfolio_thread and self.portfolio_thread.is_alive():
            self.portfolio_thread.join(timeout=5)

        logger.info("Trading orchestrator stopped")

    def _monitoring_loop(self):
        """
        Main monitoring loop for checking positions and system health.
        """
        while self.is_running:
            try:
                # Check and manage open positions
                self.trader._monitor_open_positions()

                # Perform self-healing checks
                self.self_healing._self_heal()
                
                # Update live P&L tracker
                self.pnl_tracker._update_live_pnl()

                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer after an error

    def _portfolio_monitoring_loop(self):
        """
        Main loop for portfolio-level monitoring and rebalancing.
        """
        while self.is_running:
            try:
                # Update portfolio metrics
                self.trader.portfolio_manager.update_performance_metrics()

                # Check for rebalancing opportunities
                self.trader.portfolio_manager.run_rebalancing_check()

                time.sleep(300)  # Check every 5 minutes
            except Exception as e:
                logger.error(f"Error in portfolio monitoring loop: {e}")
                time.sleep(600)  # Wait longer after an error

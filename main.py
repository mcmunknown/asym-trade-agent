#!/usr/bin/env python3
"""
SECURE MAIN APPLICATION - BULLETPROOF TRADING SYSTEM
======================================================

Institutional-grade cryptocurrency trading system with military-grade security
Replaces the vulnerable trading system with bulletproof protection against
catastrophic failures and security breaches.

SECURITY FEATURES:
- Bulletproof risk validation that CANNOT be bypassed
- Real-time security monitoring and threat detection
- Automatic system lockdown on security violations
- Comprehensive audit trails and compliance reporting
- API key rotation and secure credential management
"""

import os
import time
import logging
import signal
import sys
import threading
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import bulletproof security components
from bulletproof_trading_engine import BulletproofTradingEngine
from bulletproof_config import get_bulletproof_config, validate_leverage_limits, validate_position_size_limits
from api_key_security_manager import get_api_security_manager
from institutional_security_architecture import SecurityLevel, ThreatLevel
from data_collector import DataCollector

# Configure secure logging
def setup_secure_logging():
    """Setup secure logging with audit trail"""

    # Get log level from bulletproof config
    config_manager = get_bulletproof_config()
    log_level = config_manager.get_config_value('LOG_LEVEL', 'INFO')

    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/secure_trading_agent.log'),
            logging.FileHandler('logs/security_audit.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

setup_secure_logging()
logger = logging.getLogger(__name__)

class SecureProductionTradingAgent:
    """
    Secure production trading agent with bulletproof protection
    """

    def __init__(self, security_level: str = "MODERATE"):
        # Initialize security components
        self.config_manager = get_bulletproof_config()
        self.api_security = get_api_security_manager("SECURE_PASSWORD_CHANGE_ME_IN_PRODUCTION")

        # Initialize bulletproof trading engine
        self.trading_engine = BulletproofTradingEngine(security_level)

        # Data collection
        self.data_collector = DataCollector()

        # System state
        self.is_running = False
        self.emergency_mode = self.config_manager.get_config_value('EMERGENCY_DEEPSEEK_ONLY', False)

        # Security monitoring
        self.security_status_timer = 0
        self.last_security_check = datetime.now()

        logger.critical("🛡️ SECURE PRODUCTION TRADING AGENT INITIALIZED")
        logger.critical(f"   Security Level: {security_level}")
        logger.critical(f"   Emergency Mode: {self.emergency_mode}")
        logger.critical(f"   System Time: {datetime.now()}")

    def signal_handler(self, signum, frame):
        """Handle shutdown signals with security cleanup"""
        logger.critical(f"\n🛑 Received signal {signum}. Initiating secure shutdown...")

        # Log security event
        self.trading_engine.security_monitor.log_security_event(
            event_type="SYSTEM_SHUTDOWN",
            security_level=SecurityLevel.INFO,
            details={"signal": signum, "reason": "External signal received"},
            action_taken="Secure shutdown initiated"
        )

        self.is_running = False
        self._secure_shutdown()

    def _secure_shutdown(self):
        """Perform secure shutdown procedures"""

        logger.critical("🔒 Initiating secure shutdown procedures...")

        try:
            # Stop trading engine (closes all positions)
            self.trading_engine.stop()

            # Stop API security monitoring
            self.api_security.stop_monitoring()

            # Final security status report
            security_status = self._generate_security_report()
            logger.critical("📊 FINAL SECURITY REPORT:")
            for key, value in security_status.items():
                logger.critical(f"   {key}: {value}")

            logger.critical("✅ Secure shutdown completed")

        except Exception as e:
            logger.error(f"Error during secure shutdown: {str(e)}")

    def _system_security_check(self) -> bool:
        """Perform comprehensive system security check"""

        try:
            current_time = datetime.now()
            time_since_last_check = (current_time - self.last_security_check).total_seconds()

            # Perform security check every 5 minutes
            if time_since_last_check < 300:
                return True

            self.last_security_check = current_time

            logger.info("🔍 Performing system security check...")

            # Check configuration integrity
            config_status = self.config_manager.get_security_status()
            if config_status['system_locked']:
                logger.critical("🚨 CONFIGURATION SYSTEM LOCKED - Trading suspended")
                return False

            # Check API security status
            api_status = self.api_security.get_security_status()
            if api_status['compromised_keys'] > 0:
                logger.critical("🚨 COMPROMISED API KEYS DETECTED - Trading suspended")
                return False

            # Check trading engine security
            trading_status = self.trading_engine.get_trading_status()
            if trading_status['system_locked_down']:
                logger.critical("🚨 TRADING SYSTEM LOCKED DOWN - Trading suspended")
                return False

            # Check threat level
            if trading_status['current_threat_level'] == ThreatLevel.SEVERE.value:
                logger.critical("🚨 SEVERE THREAT LEVEL - Trading suspended")
                return False

            logger.info("✅ System security check passed")
            return True

        except Exception as e:
            logger.error(f"Error during security check: {str(e)}")
            return False

    def process_market_data(self, data_list):
        """Process market data with bulletproof security validation"""

        try:
            if not data_list:
                logger.info("No market data available")
                return

            # Security check before processing
            if not self._system_security_check():
                logger.warning("🔒 Security check failed - skipping market data processing")
                return

            logger.info(f"📊 Processing {len(data_list)} assets with bulletproof security...")

            # Process signals through secure trading engine
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                processed_signals = loop.run_until_complete(
                    self.trading_engine.process_trading_signals(data_list)
                )

                if processed_signals:
                    logger.info(f"✅ Processed {len(processed_signals)} secure trading signals")

                    # Log security event for successful processing
                    self.trading_engine.security_monitor.log_security_event(
                        event_type="SIGNALS_PROCESSED",
                        security_level=SecurityLevel.INFO,
                        details={"signal_count": len(processed_signals)},
                        action_taken="Signals processed successfully"
                    )

            finally:
                loop.close()

        except Exception as e:
            logger.error(f"Error processing market data: {str(e)}")

            # Log security event for processing error
            self.trading_engine.security_monitor.log_security_event(
                event_type="SIGNAL_PROCESSING_ERROR",
                security_level=SecurityLevel.MEDIUM,
                details={"error": str(e), "data_count": len(data_list) if data_list else 0},
                action_taken="Error logged and system continues"
            )

    def print_security_status(self):
        """Print comprehensive security status"""

        while self.is_running:
            try:
                print(f"\n{'='*80}")
                print(f"🛡️ SECURE TRADING AGENT STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*80}")

                # Security configuration status
                config_status = self.config_manager.get_security_status()
                print(f"🔒 CONFIGURATION SECURITY:")
                print(f"   System Locked: {config_status['system_locked']}")
                print(f"   Security Violations: {config_status['violation_count']}")
                print(f"   Hard Limits Enforced: {config_status['locked_parameters']} parameters")

                # API security status
                api_status = self.api_security.get_security_status()
                print(f"🔑 API SECURITY:")
                print(f"   Active Keys: {api_status['active_keys']}")
                print(f"   Compromised Keys: {api_status['compromised_keys']}")
                print(f"   Security Alerts: {api_status['security_alerts']}")

                # Trading engine security status
                trading_status = self.trading_engine.get_trading_status()
                print(f"🚀 TRADING SECURITY:")
                print(f"   System Locked Down: {trading_status['system_locked_down']}")
                print(f"   Current Threat Level: {trading_status['current_threat_level']}")
                print(f"   Active Positions: {trading_status['active_positions']}")
                print(f"   Total Trades: {trading_status['total_trades']}")

                # Get portfolio summary if available
                portfolio = self.trading_engine.get_portfolio_summary()
                if portfolio:
                    print(f"💰 PORTFOLIO:")
                    print(f"   Total Balance: ${portfolio.get('total_balance', 0):.2f}")
                    print(f"   Available: ${portfolio.get('available_balance', 0):.2f}")
                    print(f"   Active Positions: {portfolio.get('active_positions', 0)}")

                    total_invested = portfolio.get('total_invested', 0)
                    unrealized_pnl = portfolio.get('unrealized_pnl', 0)
                    if total_invested > 0:
                        pnl_change = (unrealized_pnl / total_invested) * 100
                        print(f"   Unrealized P&L: ${unrealized_pnl:.2f} ({pnl_change:+.2f}%)")

                print(f"{'='*80}\n")

                # Sleep for 15 minutes between status updates
                for _ in range(90):  # 90 * 10 seconds = 15 minutes
                    if not self.is_running:
                        break
                    time.sleep(10)

            except Exception as e:
                logger.error(f"Error printing security status: {str(e)}")
                time.sleep(60)

    def _generate_security_report(self) -> Dict:
        """Generate comprehensive security report"""

        try:
            # Get security audit report from trading engine
            audit_report = self.trading_engine.get_security_audit_report()

            # Get configuration security status
            config_status = self.config_manager.get_security_status()

            # Get API security status
            api_status = self.api_security.get_security_status()

            # Get hard limits
            hard_limits = self.config_manager.get_hard_limits()

            return {
                "audit_timestamp": audit_report['audit_timestamp'],
                "security_events": audit_report['security_events'],
                "risk_limit_violations": audit_report['risk_limit_violations'],
                "compromised_api_keys": api_status['compromised_keys'],
                "config_violations": config_status['violation_count'],
                "system_locked": config_status['system_locked'],
                "hard_limits_enforced": hard_limits,
                "total_trades_executed": audit_report['trade_executions'],
                "positions_active": audit_report['positions_active']
            }

        except Exception as e:
            logger.error(f"Error generating security report: {str(e)}")
            return {"error": str(e)}

    def run(self):
        """Main secure run loop"""

        try:
            logger.critical("🚀 Starting Secure Production Trading Agent...")

            # Initialize trading engine with security validation
            if not self.trading_engine.initialize():
                logger.critical("❌ Failed to initialize secure trading engine")
                sys.exit(1)

            # Add API keys from environment (if available)
            bybit_api_key = os.getenv("BYBIT_API_KEY")
            bybit_api_secret = os.getenv("BYBIT_API_SECRET")

            if bybit_api_key and bybit_api_secret:
                self.api_security.add_api_key(
                    exchange="BYBIT",
                    api_key=bybit_api_key,
                    api_secret=bybit_api_secret,
                    permissions=["READ", "TRADE"]
                )
                logger.info("✅ Bybit API keys secured")

            self.is_running = True

            # Set up signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)

            # Start trading engine in separate thread
            trading_thread = threading.Thread(target=self.trading_engine.start, daemon=True)
            trading_thread.start()

            # Start security status printer in separate thread
            status_thread = threading.Thread(target=self.print_security_status, daemon=True)
            status_thread.start()

            logger.critical("✅ Secure Production Trading Agent started successfully")

            # Main data collection loop with security monitoring
            while self.is_running:
                try:
                    # Security check before each cycle
                    if not self._system_security_check():
                        logger.warning("🔒 Security check failed - pausing operations")
                        time.sleep(300)  # Wait 5 minutes before retry
                        continue

                    logger.info("📊 Starting secure data collection cycle...")

                    # Collect market data
                    data_list = self.data_collector.collect_all_data()

                    if data_list:
                        logger.info(f"Collected secure data for {len(data_list)} assets")
                        self.process_market_data(data_list)
                    else:
                        logger.warning("No data collected in this cycle")

                    # Wait for next cycle (run every 30 minutes)
                    logger.info("⏰ Waiting 30 minutes for next secure analysis cycle...")

                    # Sleep in small increments to allow for graceful shutdown
                    for _ in range(180):  # 180 * 10 seconds = 30 minutes
                        if not self.is_running:
                            break

                        # Additional security check during wait
                        if self.security_status_timer % 30 == 0:  # Every 5 minutes
                            if not self._system_security_check():
                                logger.warning("🔒 Security check failed during wait - breaking cycle")
                                break

                        self.security_status_timer += 1
                        time.sleep(10)

                except Exception as e:
                    logger.error(f"Error in main secure run loop: {str(e)}")

                    # Log security event
                    self.trading_engine.security_monitor.log_security_event(
                        event_type="MAIN_LOOP_ERROR",
                        security_level=SecurityLevel.MEDIUM,
                        details={"error": str(e)},
                        action_taken="Error logged, system continues"
                    )

                    time.sleep(300)  # Wait 5 minutes before retrying

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt...")
        except Exception as e:
            logger.critical(f"Fatal error in secure main loop: {str(e)}")

            # Log critical security event
            self.trading_engine.security_monitor.log_security_event(
                event_type="FATAL_SYSTEM_ERROR",
                security_level=SecurityLevel.CRITICAL,
                details={"error": str(e)},
                action_taken="System shutdown initiated"
            )
        finally:
            self.is_running = False
            self._secure_shutdown()

def main():
    """Main entry point for secure trading system"""

    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║     🛡️ SECURE ASYMMETRIC CRYPTO TRADING AGENT v3.0            ║
    ║                                                               ║
    ║  🏛️ INSTITUTIONAL-GRADE SECURITY ARCHITECTURE                ║
    ║  🚨 BULLETPROOF RISK VALIDATION                               ║
    ║  🔔 REAL-TIME THREAT DETECTION                               ║
    ║  🔒 AUTOMATIC SYSTEM LOCKDOWN                                 ║
    ║                                                               ║
    ║  🎯 Military-Grade Protection Against:                        ║
    ║     • Leverage limit bypasses                                  ║
    ║     • Position size violations                                ║
    ║     • API key compromise                                       ║
    ║     • Configuration tampering                                  ║
    ║     • Unauthorized access                                      ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    # Show security mode
    config = get_bulletproof_config()
    emergency_mode = config.get_config_value('EMERGENCY_DEEPSEEK_ONLY', False)

    if emergency_mode:
        print("🚨 EMERGENCY MODE ACTIVATED - Maximum security protocols in place\n")
        print("   • DeepSeek V3.1-Terminus analysis only")
        print("   • Conservative risk limits enforced")
        print("   • Enhanced monitoring and alerting")
        print("   • All trades require dual validation\n")
    else:
        print("🛡️ STANDARD SECURITY MODE - Institutional-grade protection active\n")
        print("   • Multi-model consensus analysis")
        print("   • Bulletproof risk validation")
        print("   • Real-time security monitoring")
        print("   • Comprehensive audit trails\n")

    # Show hard security limits
    hard_limits = config.get_hard_limits()
    print("🔒 HARD SECURITY LIMITS (CANNOT BE EXCEEDED):")
    print(f"   • Maximum Leverage: {hard_limits['max_leverage_hard_limit']}x")
    print(f"   • Max Position Size: {hard_limits['max_position_size_pct_hard_limit']}%")
    print(f"   • Max Total Exposure: {hard_limits['max_total_exposure_pct_hard_limit']}%")
    print(f"   • Emergency Stop Loss: {hard_limits['emergency_stop_loss_pct_hard_limit']}%\n")

    print("⚠️  ALL SECURITY VIOLATIONS TRIGGER IMMEDIATE SYSTEM LOCKDOWN")
    print("📊 Comprehensive audit trails maintained for compliance")
    print("🔑 API keys automatically rotated every 24 hours\n")

    print("Press Ctrl+C to initiate secure shutdown\n")

    try:
        # Determine security level based on configuration
        security_level = "CONSERVATIVE" if emergency_mode else "MODERATE"

        agent = SecureProductionTradingAgent(security_level)
        agent.run()

    except Exception as e:
        logger.critical(f"Application startup error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
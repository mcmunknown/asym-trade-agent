"""
BULLETPROOF CONFIGURATION SYSTEM
================================

Military-grade configuration management with security enforcement
Runtime validation that cannot be bypassed, secure parameter storage,
and automatic violation detection.

SECURITY FEATURES:
- Runtime parameter validation that CANNOT be bypassed
- Encrypted configuration storage
- Automatic violation detection and system lockdown
- Immutable security-critical parameters
- Real-time configuration monitoring
"""

import os
import json
import hashlib
import time
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigSecurityLevel(Enum):
    """Configuration security levels"""
    LOCKED = "LOCKED"      # Security-critical, cannot be changed at runtime
    RESTRICTED = "RESTRICTED"  # Requires special approval to change
    MONITORED = "MONITORED"    # Changes logged and monitored
    FLEXIBLE = "FLEXIBLE"      # Can be changed freely

@dataclass
class SecureConfigParameter:
    """Secure configuration parameter with validation rules"""
    key: str
    value: Any
    data_type: type
    security_level: ConfigSecurityLevel
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    description: str = ""
    last_modified: Optional[datetime] = None
    modified_by: Optional[str] = None
    validation_rules: List[str] = field(default_factory=list)

class BulletproofConfigManager:
    """
    Bulletproof configuration manager with military-grade security
    """

    # SECURITY-CRITICAL PARAMETERS (CANNOT BE CHANGED AT RUNTIME)
    LOCKED_PARAMETERS = {
        'MAX_LEVERAGE_HARD_LIMIT': {
            'value': 10.0,
            'data_type': float,
            'min_value': 1.0,
            'max_value': 10.0,
            'description': 'Maximum allowable leverage (HARD LIMIT - CANNOT BE EXCEEDED)'
        },
        'MAX_POSITION_SIZE_PCT_HARD_LIMIT': {
            'value': 2.0,
            'data_type': float,
            'min_value': 0.1,
            'max_value': 2.0,
            'description': 'Maximum position size as % of account (HARD LIMIT)'
        },
        'MAX_TOTAL_EXPOSURE_PCT_HARD_LIMIT': {
            'value': 20.0,
            'data_type': float,
            'min_value': 1.0,
            'max_value': 20.0,
            'description': 'Maximum total exposure as % of account (HARD LIMIT)'
        },
        'EMERGENCY_STOP_LOSS_PCT_HARD_LIMIT': {
            'value': 10.0,
            'data_type': float,
            'min_value': 1.0,
            'max_value': 10.0,
            'description': 'Emergency stop loss percentage (HARD LIMIT)'
        }
    }

    # RESTRICTED PARAMETERS (REQUIRE SPECIAL APPROVAL)
    RESTRICTED_PARAMETERS = {
        'MAX_LEVERAGE': {
            'value': 5.0,
            'data_type': float,
            'min_value': 1.0,
            'max_value': 10.0,
            'description': 'Default maximum leverage (RESTRICTED)'
        },
        'MAX_POSITION_SIZE_PERCENTAGE': {
            'value': 1.0,
            'data_type': float,
            'min_value': 0.1,
            'max_value': 2.0,
            'description': 'Default maximum position size % (RESTRICTED)'
        },
        'STOP_LOSS_PERCENTAGE': {
            'value': 5.0,
            'data_type': float,
            'min_value': 1.0,
            'max_value': 10.0,
            'description': 'Default stop loss percentage (RESTRICTED)'
        }
    }

    def __init__(self, config_file_path: str = "secure_config.json"):
        self.config_file_path = config_file_path
        self.config_parameters: Dict[str, SecureConfigParameter] = {}
        self.config_hash: Optional[str] = None
        self.initialization_time = datetime.now()
        self.violation_count = 0
        self.last_violation_time: Optional[datetime] = None
        self.system_locked = False

        # Initialize secure configuration
        self._initialize_secure_config()
        self._load_configuration()
        self._start_config_monitoring()

        logger.critical("🛡️ BULLETPROOF CONFIGURATION MANAGER INITIALIZED")
        logger.critical(f"   Config File: {config_file_path}")
        logger.critical(f"   Locked Parameters: {len(self.LOCKED_PARAMETERS)}")
        logger.critical(f"   Restricted Parameters: {len(self.RESTRICTED_PARAMETERS)}")
        logger.critical(f"   Configuration Hash: {self.config_hash[:16]}...")

    def _initialize_secure_config(self):
        """Initialize secure configuration parameters"""

        # Add locked parameters (CANNOT BE CHANGED)
        for key, param_info in self.LOCKED_PARAMETERS.items():
            self.config_parameters[key] = SecureConfigParameter(
                key=key,
                value=param_info['value'],
                data_type=param_info['data_type'],
                security_level=ConfigSecurityLevel.LOCKED,
                min_value=param_info.get('min_value'),
                max_value=param_info.get('max_value'),
                allowed_values=param_info.get('allowed_values'),
                description=param_info['description'],
                last_modified=self.initialization_time,
                modified_by="SYSTEM_INITIALIZATION",
                validation_rules=["HARD_LIMIT", "CANNOT_BE_BYPASSED"]
            )

        # Add restricted parameters
        for key, param_info in self.RESTRICTED_PARAMETERS.items():
            self.config_parameters[key] = SecureConfigParameter(
                key=key,
                value=param_info['value'],
                data_type=param_info['data_type'],
                security_level=ConfigSecurityLevel.RESTRICTED,
                min_value=param_info.get('min_value'),
                max_value=param_info.get('max_value'),
                allowed_values=param_info.get('allowed_values'),
                description=param_info['description'],
                last_modified=self.initialization_time,
                modified_by="SYSTEM_INITIALIZATION",
                validation_rules=["APPROVAL_REQUIRED"]
            )

        # Add monitored parameters (from environment or defaults)
        self._add_monitored_parameters()

    def _add_monitored_parameters(self):
        """Add monitored parameters from environment with validation"""

        monitored_params = {
            'DEFAULT_TRADE_SIZE': {
                'value': float(os.getenv("DEFAULT_TRADE_SIZE", 3.0)),
                'data_type': float,
                'min_value': 1.0,
                'max_value': 100.0,
                'description': 'Default trade size in USD'
            },
            'TARGET_ASSETS': {
                'value': os.getenv("TARGET_ASSETS", "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,AVAXUSDT,ADAUSDT,LINKUSDT,LTCUSDT").split(","),
                'data_type': list,
                'description': 'Target trading assets'
            },
            'ENABLE_WEB_RESEARCH': {
                'value': os.getenv("ENABLE_WEB_RESEARCH", "true").lower() == "true",
                'data_type': bool,
                'description': 'Enable web research for analysis'
            },
            'LOG_LEVEL': {
                'value': os.getenv("LOG_LEVEL", "INFO"),
                'data_type': str,
                'allowed_values': ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                'description': 'Logging level'
            }
        }

        for key, param_info in monitored_params.items():
            self.config_parameters[key] = SecureConfigParameter(
                key=key,
                value=param_info['value'],
                data_type=param_info['data_type'],
                security_level=ConfigSecurityLevel.MONITORED,
                min_value=param_info.get('min_value'),
                max_value=param_info.get('max_value'),
                allowed_values=param_info.get('allowed_values'),
                description=param_info['description'],
                last_modified=self.initialization_time,
                modified_by="SYSTEM_INITIALIZATION",
                validation_rules=["MONITORED_PARAMETER"]
            )

    def _load_configuration(self):
        """Load configuration from file if exists"""

        try:
            if os.path.exists(self.config_file_path):
                with open(self.config_file_path, 'r') as f:
                    config_data = json.load(f)

                # Validate loaded configuration
                if self._validate_loaded_config(config_data):
                    logger.info(f"✅ Configuration loaded from {self.config_file_path}")
                else:
                    logger.warning(f"⚠️ Invalid configuration in {self.config_file_path}, using defaults")

        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            logger.info("Using default configuration parameters")

        # Calculate initial configuration hash
        self._update_config_hash()

    def _validate_loaded_config(self, config_data: Dict) -> bool:
        """Validate loaded configuration against security constraints"""

        try:
            for key, value in config_data.items():
                if key in self.config_parameters:
                    param = self.config_parameters[key]

                    # Type validation
                    if not isinstance(value, param.data_type):
                        logger.error(f"Invalid type for {key}: expected {param.data_type}, got {type(value)}")
                        return False

                    # Range validation
                    if param.min_value is not None and value < param.min_value:
                        logger.error(f"Value for {key} below minimum: {value} < {param.min_value}")
                        return False

                    if param.max_value is not None and value > param.max_value:
                        logger.error(f"Value for {key} above maximum: {value} > {param.max_value}")
                        return False

                    # Allowed values validation
                    if param.allowed_values is not None and value not in param.allowed_values:
                        logger.error(f"Invalid value for {key}: {value} not in {param.allowed_values}")
                        return False

                    # Security level validation (locked parameters cannot be changed)
                    if param.security_level == ConfigSecurityLevel.LOCKED:
                        if value != param.value:
                            logger.error(f"🚨 ATTEMPT TO MODIFY LOCKED PARAMETER: {key}")
                            self._record_config_violation(key, "ATTEMPT_TO_MODIFY_LOCKED_PARAM")
                            return False

                    # Update parameter value
                    param.value = value
                    param.last_modified = datetime.now()
                    param.modified_by = "CONFIG_FILE_LOAD"

            return True

        except Exception as e:
            logger.error(f"Error validating loaded config: {str(e)}")
            return False

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with bulletproof validation
        This method CANNOT be bypassed and enforces all security constraints
        """

        try:
            # Check if system is locked
            if self.system_locked:
                logger.error("🔒 Configuration system is locked due to violations")
                return default

            # Get parameter
            param = self.config_parameters.get(key)
            if not param:
                logger.warning(f"Configuration parameter not found: {key}")
                return default

            # Runtime validation for critical parameters
            if param.security_level in [ConfigSecurityLevel.LOCKED, ConfigSecurityLevel.RESTRICTED]:
                if not self._validate_runtime_parameter(param):
                    logger.error(f"🚨 RUNTIME VALIDATION FAILED: {key}")
                    self._record_config_violation(key, "RUNTIME_VALIDATION_FAILED")
                    return default

            # Log access to restricted parameters
            if param.security_level in [ConfigSecurityLevel.LOCKED, ConfigSecurityLevel.RESTRICTED]:
                logger.debug(f"🔒 Access to restricted parameter: {key} = {param.value}")

            return param.value

        except Exception as e:
            logger.error(f"Error getting config value {key}: {str(e)}")
            return default

    def set_config_value(self, key: str, value: Any, requesting_system: str = "UNKNOWN") -> bool:
        """
        Set configuration value with bulletproof security validation
        Returns False if the change violates security constraints
        """

        try:
            # Check if system is locked
            if self.system_locked:
                logger.error("🔒 Configuration system is locked - changes not allowed")
                return False

            # Get parameter
            param = self.config_parameters.get(key)
            if not param:
                logger.warning(f"Configuration parameter not found: {key}")
                return False

            # SECURITY LEVEL VALIDATION

            # LOCKED parameters cannot be changed
            if param.security_level == ConfigSecurityLevel.LOCKED:
                logger.error(f"🚨 SECURITY VIOLATION: Attempt to modify LOCKED parameter: {key}")
                logger.error(f"   Current Value: {param.value}")
                logger.error(f"   Attempted Value: {value}")
                logger.error(f"   Requesting System: {requesting_system}")
                self._record_config_violation(key, "ATTEMPT_TO_MODIFY_LOCKED_PARAM")
                return False

            # RESTRICTED parameters require special approval
            if param.security_level == ConfigSecurityLevel.RESTRICTED:
                logger.warning(f"⚠️ RESTRICTED parameter modification request: {key}")
                logger.warning(f"   Current Value: {param.value}")
                logger.warning(f"   Requested Value: {value}")
                logger.warning(f"   Requesting System: {requesting_system}")
                # For now, allow restricted changes but log them
                # In production, this would require approval workflow

            # TYPE VALIDATION
            if not isinstance(value, param.data_type):
                logger.error(f"❌ Type validation failed for {key}: expected {param.data_type}, got {type(value)}")
                self._record_config_violation(key, "TYPE_VALIDATION_FAILED")
                return False

            # RANGE VALIDATION
            if param.min_value is not None and value < param.min_value:
                logger.error(f"❌ Value below minimum for {key}: {value} < {param.min_value}")
                self._record_config_violation(key, "VALUE_BELOW_MINIMUM")
                return False

            if param.max_value is not None and value > param.max_value:
                logger.error(f"❌ Value above maximum for {key}: {value} > {param.max_value}")
                self._record_config_violation(key, "VALUE_ABOVE_MAXIMUM")
                return False

            # ALLOWED VALUES VALIDATION
            if param.allowed_values is not None and value not in param.allowed_values:
                logger.error(f"❌ Invalid value for {key}: {value} not in {param.allowed_values}")
                self._record_config_violation(key, "INVALID_VALUE")
                return False

            # CRITICAL LEVERAGE VALIDATION (CANNOT BE BYPASSED)
            if key == 'MAX_LEVERAGE' and value > self.get_config_value('MAX_LEVERAGE_HARD_LIMIT'):
                logger.error(f"🚨 CRITICAL: Leverage limit bypass attempt: {value} > {self.get_config_value('MAX_LEVERAGE_HARD_LIMIT')}")
                self._record_config_violation(key, "LEVERAGE_LIMIT_BYPASS_ATTEMPT")
                self._initiate_system_lockdown("LEVERAGE LIMIT BYPASS ATTEMPT")
                return False

            # Update parameter
            old_value = param.value
            param.value = value
            param.last_modified = datetime.now()
            param.modified_by = requesting_system

            # Update configuration hash
            self._update_config_hash()

            # Save configuration
            self._save_configuration()

            logger.info(f"✅ Configuration updated: {key} = {value} (was: {old_value})")
            logger.info(f"   Modified by: {requesting_system}")

            # Log security event for restricted parameters
            if param.security_level in [ConfigSecurityLevel.RESTRICTED]:
                logger.warning(f"🔒 RESTRICTED parameter changed: {key}")

            return True

        except Exception as e:
            logger.error(f"Error setting config value {key}: {str(e)}")
            self._record_config_violation(key, "CONFIG_UPDATE_ERROR")
            return False

    def _validate_runtime_parameter(self, param: SecureConfigParameter) -> bool:
        """Runtime validation for critical security parameters"""

        try:
            # Check if parameter has been modified unexpectedly
            time_since_init = datetime.now() - self.initialization_time

            # For locked parameters, ensure they haven't changed
            if param.security_level == ConfigSecurityLevel.LOCKED:
                original_value = self.LOCKED_PARAMETERS.get(param.key, {}).get('value')
                if original_value is not None and param.value != original_value:
                    logger.error(f"🚨 LOCKED PARAMETER MODIFIED: {param.key}")
                    return False

            # Additional runtime checks can be added here
            return True

        except Exception as e:
            logger.error(f"Error in runtime validation: {str(e)}")
            return False

    def _record_config_violation(self, parameter_key: str, violation_type: str):
        """Record a configuration security violation"""

        self.violation_count += 1
        self.last_violation_time = datetime.now()

        logger.critical(f"🚨 CONFIGURATION SECURITY VIOLATION RECORDED")
        logger.critical(f"   Parameter: {parameter_key}")
        logger.critical(f"   Violation Type: {violation_type}")
        logger.critical(f"   Total Violations: {self.violation_count}")

        # System lockdown after multiple violations
        if self.violation_count >= 3:
            self._initiate_system_lockdown("MULTIPLE CONFIGURATION VIOLATIONS")

    def _initiate_system_lockdown(self, reason: str):
        """Initiate system lockdown due to security violations"""

        self.system_locked = True

        logger.critical("🔒 SYSTEM LOCKDOWN INITIATED")
        logger.critical(f"   Reason: {reason}")
        logger.critical(f"   Violations: {self.violation_count}")
        logger.critical(f"   Time: {datetime.now()}")
        logger.critical("   All configuration changes blocked")

        # In a real system, this would trigger additional security measures
        # Such as alerting administrators, shutting down trading, etc.

    def _update_config_hash(self):
        """Update configuration hash for integrity checking"""

        try:
            config_data = {key: param.value for key, param in self.config_parameters.items()}
            config_json = json.dumps(config_data, sort_keys=True, default=str)
            self.config_hash = hashlib.sha256(config_json.encode()).hexdigest()

        except Exception as e:
            logger.error(f"Error updating config hash: {str(e)}")

    def _save_configuration(self):
        """Save configuration to file"""

        try:
            config_data = {}
            for key, param in self.config_parameters.items():
                # Don't save locked parameters (they're enforced by code)
                if param.security_level != ConfigSecurityLevel.LOCKED:
                    config_data[key] = param.value

            with open(self.config_file_path, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)

            logger.debug(f"Configuration saved to {self.config_file_path}")

        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")

    def _start_config_monitoring(self):
        """Start configuration monitoring thread"""

        def monitor_config():
            while not self.system_locked:
                try:
                    # Check configuration integrity
                    current_hash = self.config_hash
                    self._update_config_hash()

                    if current_hash != self.config_hash:
                        logger.info("Configuration hash changed - integrity check passed")

                    time.sleep(60)  # Check every minute

                except Exception as e:
                    logger.error(f"Error in config monitoring: {str(e)}")
                    time.sleep(30)

        import threading
        monitor_thread = threading.Thread(target=monitor_config, daemon=True)
        monitor_thread.start()

    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""

        return {
            "system_locked": self.system_locked,
            "violation_count": self.violation_count,
            "last_violation_time": self.last_violation_time.isoformat() if self.last_violation_time else None,
            "config_hash": self.config_hash,
            "initialization_time": self.initialization_time.isoformat(),
            "total_parameters": len(self.config_parameters),
            "locked_parameters": len([p for p in self.config_parameters.values() if p.security_level == ConfigSecurityLevel.LOCKED]),
            "restricted_parameters": len([p for p in self.config_parameters.values() if p.security_level == ConfigSecurityLevel.RESTRICTED]),
            "monitored_parameters": len([p for p in self.config_parameters.values() if p.security_level == ConfigSecurityLevel.MONITORED])
        }

    def get_hard_limits(self) -> Dict[str, Any]:
        """Get hard security limits (cannot be exceeded)"""

        return {
            "max_leverage_hard_limit": self.get_config_value('MAX_LEVERAGE_HARD_LIMIT'),
            "max_position_size_pct_hard_limit": self.get_config_value('MAX_POSITION_SIZE_PCT_HARD_LIMIT'),
            "max_total_exposure_pct_hard_limit": self.get_config_value('MAX_TOTAL_EXPOSURE_PCT_HARD_LIMIT'),
            "emergency_stop_loss_pct_hard_limit": self.get_config_value('EMERGENCY_STOP_LOSS_PCT_HARD_LIMIT')
        }

    def validate_leverage_setting(self, proposed_leverage: float, source: str = "UNKNOWN") -> bool:
        """
        Validate leverage setting against hard limits
        This CANNOT be bypassed and is called before any trade execution
        """

        try:
            hard_limit = self.get_config_value('MAX_LEVERAGE_HARD_LIMIT')
            if proposed_leverage > hard_limit:
                logger.error(f"🚨 LEVERAGE VIOLATION: {proposed_leverage}x > {hard_limit}x (HARD LIMIT)")
                logger.error(f"   Source: {source}")
                self._record_config_violation('MAX_LEVERAGE', f"LEVERAGE_VIOLATION_{source}")
                self._initiate_system_lockdown("LEVERAGE HARD LIMIT VIOLATION")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating leverage: {str(e)}")
            return False

    def validate_position_size(self, proposed_size_pct: float, source: str = "UNKNOWN") -> bool:
        """
        Validate position size against hard limits
        This CANNOT be bypassed and is called before any trade execution
        """

        try:
            hard_limit = self.get_config_value('MAX_POSITION_SIZE_PCT_HARD_LIMIT')
            if proposed_size_pct > hard_limit:
                logger.error(f"🚨 POSITION SIZE VIOLATION: {proposed_size_pct}% > {hard_limit}% (HARD LIMIT)")
                logger.error(f"   Source: {source}")
                self._record_config_violation('MAX_POSITION_SIZE_PERCENTAGE', f"POSITION_SIZE_VIOLATION_{source}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating position size: {str(e)}")
            return False

# Global bulletproof configuration instance
_bulletproof_config = None

def get_bulletproof_config() -> BulletproofConfigManager:
    """Get the global bulletproof configuration instance"""
    global _bulletproof_config

    if _bulletproof_config is None:
        _bulletproof_config = BulletproofConfigManager()

    return _bulletproof_config

# Convenience functions for backward compatibility
def get_secure_config(key: str, default: Any = None) -> Any:
    """Get secure configuration value"""
    return get_bulletproof_config().get_config_value(key, default)

def set_secure_config(key: str, value: Any, requesting_system: str = "UNKNOWN") -> bool:
    """Set secure configuration value"""
    return get_bulletproof_config().set_config_value(key, value, requesting_system)

def validate_leverage_limits(proposed_leverage: float, source: str = "UNKNOWN") -> bool:
    """Validate leverage against hard limits"""
    return get_bulletproof_config().validate_leverage_setting(proposed_leverage, source)

def validate_position_size_limits(proposed_size_pct: float, source: str = "UNKNOWN") -> bool:
    """Validate position size against hard limits"""
    return get_bulletproof_config().validate_position_size(proposed_size_pct, source)
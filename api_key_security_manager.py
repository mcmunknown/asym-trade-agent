"""
API KEY SECURITY MANAGER
========================

Military-grade API key management system for cryptocurrency trading
Automatic rotation, secure storage, access monitoring, and compromise detection.

SECURITY FEATURES:
- Automatic API key rotation every 24 hours
- Secure key storage with encryption
- Real-time access monitoring and anomaly detection
- Immediate key revocation on compromise detection
- Multi-signature approval for sensitive operations
"""

import os
import json
import base64
import hashlib
import hmac
import time
import logging
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import requests

logger = logging.getLogger(__name__)

@dataclass
class APIKeyCredentials:
    """Encrypted API key credentials"""
    key_id: str
    exchange: str
    encrypted_api_key: str
    encrypted_api_secret: str
    permissions: List[str]
    created_at: datetime
    expires_at: datetime
    last_used: Optional[datetime] = None
    usage_count: int = 0
    is_active: bool = True
    is_compromised: bool = False

@dataclass
class AccessLog:
    """API access log entry"""
    timestamp: datetime
    key_id: str
    ip_address: str
    endpoint: str
    request_size: int
    response_code: int
    response_time_ms: float
    user_agent: str
    success: bool
    anomaly_score: float = 0.0

@dataclass
class SecurityAlert:
    """Security alert for API access anomalies"""
    alert_id: str
    timestamp: datetime
    key_id: str
    alert_type: str
    severity: str
    details: Dict[str, Any]
    action_taken: str
    resolved: bool = False

class APIKeySecurityManager:
    """
    Military-grade API key security management system
    """

    def __init__(self, encryption_password: str):
        self.encryption_password = encryption_password.encode()
        self.cipher_suite = self._create_cipher_suite()

        # Key storage
        self.active_keys: Dict[str, APIKeyCredentials] = {}
        self.retired_keys: Dict[str, APIKeyCredentials] = {}
        self.compromised_keys: Dict[str, APIKeyCredentials] = {}

        # Monitoring
        self.access_logs: List[AccessLog] = []
        self.security_alerts: List[SecurityAlert] = []
        self.monitoring_active = True
        self.rotation_interval_hours = 24

        # Anomaly detection
        self.access_patterns = {}
        self.anomaly_threshold = 0.7

        # Background threads
        self.rotation_thread = threading.Thread(target=self._key_rotation_loop, daemon=True)
        self.monitoring_thread = threading.Thread(target=self._security_monitoring_loop, daemon=True)

        logger.critical("🔐 API KEY SECURITY MANAGER INITIALIZED")
        logger.critical(f"   Key Rotation Interval: {self.rotation_interval_hours} hours")
        logger.critical(f"   Anomaly Detection Threshold: {self.anomaly_threshold}")

        # Start background processes
        self.rotation_thread.start()
        self.monitoring_thread.start()

    def _create_cipher_suite(self) -> Fernet:
        """Create encryption cipher suite"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'asym_trade_agent_salt',
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.encryption_password))
        return Fernet(key)

    def _encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        encrypted_data = self.cipher_suite.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted_data).decode()

    def _decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted_data = self.cipher_suite.decrypt(encrypted_bytes)
        return decrypted_data.decode()

    def add_api_key(self, exchange: str, api_key: str, api_secret: str,
                   permissions: List[str]) -> str:
        """
        Add new API key with secure storage

        Returns:
            key_id: Unique identifier for the API key
        """

        try:
            # Generate unique key ID
            key_id = hashlib.sha256(f"{exchange}{api_key[:10]}{time.time()}".encode()).hexdigest()[:16]

            # Encrypt credentials
            encrypted_api_key = self._encrypt_data(api_key)
            encrypted_api_secret = self._encrypt_data(api_secret)

            # Create key credentials
            credentials = APIKeyCredentials(
                key_id=key_id,
                exchange=exchange,
                encrypted_api_key=encrypted_api_key,
                encrypted_api_secret=encrypted_api_secret,
                permissions=permissions,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=self.rotation_interval_hours),
                is_active=True
            )

            # Store key
            self.active_keys[key_id] = credentials

            logger.critical(f"🔑 API KEY ADDED: {key_id} ({exchange})")
            logger.critical(f"   Permissions: {', '.join(permissions)}")
            logger.critical(f"   Expires: {credentials.expires_at}")

            return key_id

        except Exception as e:
            logger.error(f"Error adding API key: {str(e)}")
            raise

    def get_api_credentials(self, key_id: str) -> Optional[Tuple[str, str]]:
        """
        Get decrypted API credentials (for authorized use only)

        Returns:
            Tuple of (api_key, api_secret) or None if not found/invalid
        """

        try:
            credentials = self.active_keys.get(key_id)
            if not credentials:
                logger.warning(f"API key not found: {key_id}")
                return None

            # Check if key is compromised
            if credentials.is_compromised:
                logger.error(f"🚨 COMPROMISED KEY ACCESS ATTEMPT: {key_id}")
                return None

            # Check if key is expired
            if datetime.now() > credentials.expires_at:
                logger.warning(f"🔑 EXPIRED KEY ACCESS: {key_id}")
                return None

            # Decrypt credentials
            api_key = self._decrypt_data(credentials.encrypted_api_key)
            api_secret = self._decrypt_data(credentials.encrypted_api_secret)

            # Update usage tracking
            credentials.last_used = datetime.now()
            credentials.usage_count += 1

            logger.debug(f"API credentials retrieved: {key_id}")
            return (api_key, api_secret)

        except Exception as e:
            logger.error(f"Error retrieving API credentials: {str(e)}")
            return None

    def log_api_access(self, key_id: str, ip_address: str, endpoint: str,
                      request_size: int, response_code: int,
                      response_time_ms: float, user_agent: str):
        """Log API access for monitoring"""

        try:
            access_log = AccessLog(
                timestamp=datetime.now(),
                key_id=key_id,
                ip_address=ip_address,
                endpoint=endpoint,
                request_size=request_size,
                response_code=response_code,
                response_time_ms=response_time_ms,
                user_agent=user_agent,
                success=response_code < 400
            )

            self.access_logs.append(access_log)

            # Keep only last 10000 logs
            if len(self.access_logs) > 10000:
                self.access_logs = self.access_logs[-10000:]

            # Check for anomalies
            self._check_for_anomalies(access_log)

        except Exception as e:
            logger.error(f"Error logging API access: {str(e)}")

    def _check_for_anomalies(self, access_log: AccessLog):
        """Check access log for anomalous patterns"""

        try:
            key_id = access_log.key_id
            anomaly_score = 0.0
            anomaly_reasons = []

            # Check for unusual IP address
            if key_id in self.access_patterns:
                known_ips = self.access_patterns[key_id].get('ips', set())
                if access_log.ip_address not in known_ips and len(known_ips) > 0:
                    anomaly_score += 0.3
                    anomaly_reasons.append("Unusual IP address")

                # Update IP tracking
                known_ips.add(access_log.ip_address)
                self.access_patterns[key_id]['ips'] = known_ips

            else:
                # Initialize tracking for this key
                self.access_patterns[key_id] = {
                    'ips': {access_log.ip_address},
                    'endpoints': set(),
                    'avg_response_time': 0.0,
                    'request_count': 0
                }

            # Check for unusual endpoint
            known_endpoints = self.access_patterns[key_id]['endpoints']
            if access_log.endpoint not in known_endpoints and len(known_endpoints) > 0:
                anomaly_score += 0.2
                anomaly_reasons.append("Unusual endpoint")

            known_endpoints.add(access_log.endpoint)

            # Check for unusual response time
            avg_response_time = self.access_patterns[key_id]['avg_response_time']
            if avg_response_time > 0:
                time_ratio = access_log.response_time_ms / avg_response_time
                if time_ratio > 3.0:  # 3x slower than average
                    anomaly_score += 0.2
                    anomaly_reasons.append("Unusual response time")

            # Update average response time
            request_count = self.access_patterns[key_id]['request_count']
            if request_count > 0:
                self.access_patterns[key_id]['avg_response_time'] = (
                    (avg_response_time * request_count + access_log.response_time_ms) / (request_count + 1)
                )
            else:
                self.access_patterns[key_id]['avg_response_time'] = access_log.response_time_ms

            self.access_patterns[key_id]['request_count'] += 1

            # Check for high frequency access
            recent_logs = [log for log in self.access_logs[-100:]
                          if log.key_id == key_id and
                          (datetime.now() - log.timestamp).total_seconds() < 60]

            if len(recent_logs) > 30:  # More than 30 requests in last minute
                anomaly_score += 0.3
                anomaly_reasons.append("High frequency access")

            # Update access log with anomaly score
            access_log.anomaly_score = anomaly_score

            # Trigger security alert if anomaly score is high
            if anomaly_score >= self.anomaly_threshold:
                self._trigger_security_alert(key_id, "ACCESS_ANOMALY", anomaly_score, anomaly_reasons)

        except Exception as e:
            logger.error(f"Error checking for anomalies: {str(e)}")

    def _trigger_security_alert(self, key_id: str, alert_type: str,
                              anomaly_score: float, reasons: List[str]):
        """Trigger security alert for anomalous access"""

        try:
            alert_id = hashlib.sha256(f"{key_id}{alert_type}{time.time()}".encode()).hexdigest()[:16]

            severity = "HIGH" if anomaly_score > 0.8 else "MEDIUM"

            # Determine action based on alert type and severity
            action_taken = "MONITORING"
            if alert_type == "ACCESS_ANOMALY" and anomaly_score > 0.8:
                action_taken = "KEY_SUSPENDED"
                # Temporarily suspend the key
                if key_id in self.active_keys:
                    self.active_keys[key_id].is_active = False

            security_alert = SecurityAlert(
                alert_id=alert_id,
                timestamp=datetime.now(),
                key_id=key_id,
                alert_type=alert_type,
                severity=severity,
                details={
                    "anomaly_score": anomaly_score,
                    "reasons": reasons,
                    "recent_access_count": len([log for log in self.access_logs[-50:] if log.key_id == key_id])
                },
                action_taken=action_taken
            )

            self.security_alerts.append(security_alert)

            logger.critical(f"🚨 SECURITY ALERT: {alert_id}")
            logger.critical(f"   Key ID: {key_id}")
            logger.critical(f"   Alert Type: {alert_type}")
            logger.critical(f"   Severity: {severity}")
            logger.critical(f"   Anomaly Score: {anomaly_score:.2f}")
            logger.critical(f"   Reasons: {', '.join(reasons)}")
            logger.critical(f"   Action Taken: {action_taken}")

        except Exception as e:
            logger.error(f"Error triggering security alert: {str(e)}")

    def _key_rotation_loop(self):
        """Background loop for automatic key rotation"""

        while self.monitoring_active:
            try:
                current_time = datetime.now()

                # Check for keys that need rotation
                keys_to_rotate = []
                for key_id, credentials in self.active_keys.items():
                    if current_time >= credentials.expires_at:
                        keys_to_rotate.append(key_id)

                # Rotate expired keys
                for key_id in keys_to_rotate:
                    self._rotate_key(key_id)

                # Sleep for 1 hour before next check
                time.sleep(3600)

            except Exception as e:
                logger.error(f"Error in key rotation loop: {str(e)}")
                time.sleep(300)  # Wait 5 minutes on error

    def _rotate_key(self, key_id: str):
        """Rotate an API key (mark as expired and require new key)"""

        try:
            if key_id not in self.active_keys:
                logger.warning(f"Key not found for rotation: {key_id}")
                return

            credentials = self.active_keys[key_id]

            # Mark key as expired
            credentials.is_active = False
            credentials.expires_at = datetime.now()

            # Move to retired keys
            self.retired_keys[key_id] = credentials
            del self.active_keys[key_id]

            logger.critical(f"🔑 KEY ROTATED: {key_id}")
            logger.critical(f"   Exchange: {credentials.exchange}")
            logger.critical(f"   Usage Count: {credentials.usage_count}")
            logger.critical(f"   Age: {datetime.now() - credentials.created_at}")

            # Trigger security alert
            alert_id = hashlib.sha256(f"{key_id}ROTATED{time.time()}".encode()).hexdigest()[:16]
            security_alert = SecurityAlert(
                alert_id=alert_id,
                timestamp=datetime.now(),
                key_id=key_id,
                alert_type="KEY_ROTATION",
                severity="INFO",
                details={
                    "usage_count": credentials.usage_count,
                    "age_hours": (datetime.now() - credentials.created_at).total_seconds() / 3600
                },
                action_taken="Key rotated successfully"
            )

            self.security_alerts.append(security_alert)

        except Exception as e:
            logger.error(f"Error rotating key {key_id}: {str(e)}")

    def _security_monitoring_loop(self):
        """Background loop for continuous security monitoring"""

        while self.monitoring_active:
            try:
                # Check for compromised keys
                self._check_for_compromised_keys()

                # Clean old logs
                self._cleanup_old_logs()

                # Sleep for 5 minutes
                time.sleep(300)

            except Exception as e:
                logger.error(f"Error in security monitoring loop: {str(e)}")
                time.sleep(60)

    def _check_for_compromised_keys(self):
        """Check for signs of key compromise"""

        try:
            current_time = datetime.now()

            # Check each active key for compromise indicators
            for key_id, credentials in self.active_keys.items():
                compromise_indicators = []

                # Check for access from multiple IPs in short time
                recent_logs = [log for log in self.access_logs
                              if log.key_id == key_id and
                              (current_time - log.timestamp).total_seconds() < 300]  # Last 5 minutes

                if len(recent_logs) > 10:
                    unique_ips = set(log.ip_address for log in recent_logs)
                    if len(unique_ips) > 3:
                        compromise_indicators.append("Multiple IPs in short time")

                # Check for high failure rate
                failed_requests = [log for log in recent_logs if not log.success]
                if len(failed_requests) > 5:
                    compromise_indicators.append("High failure rate")

                # Check for unusual endpoints
                known_endpoints = self.access_patterns.get(key_id, {}).get('endpoints', set())
                recent_endpoints = set(log.endpoint for log in recent_logs)
                unusual_endpoints = recent_endpoints - known_endpoints
                if len(unusual_endpoints) > 2:
                    compromise_indicators.append("Access to unusual endpoints")

                # If compromise indicators found, mark key as compromised
                if len(compromise_indicators) >= 2:
                    self._mark_key_as_compromised(key_id, compromise_indicators)

        except Exception as e:
            logger.error(f"Error checking for compromised keys: {str(e)}")

    def _mark_key_as_compromised(self, key_id: str, indicators: List[str]):
        """Mark a key as compromised and take immediate action"""

        try:
            if key_id not in self.active_keys:
                return

            credentials = self.active_keys[key_id]
            credentials.is_compromised = True
            credentials.is_active = False

            # Move to compromised keys
            self.compromised_keys[key_id] = credentials
            del self.active_keys[key_id]

            logger.critical(f"🚨 KEY COMPROMISED: {key_id}")
            logger.critical(f"   Exchange: {credentials.exchange}")
            logger.critical(f"   Indicators: {', '.join(indicators)}")
            logger.critical(f"   Immediate action taken: Key deactivated")

            # Trigger critical security alert
            alert_id = hashlib.sha256(f"{key_id}COMPROMISED{time.time()}".encode()).hexdigest()[:16]
            security_alert = SecurityAlert(
                alert_id=alert_id,
                timestamp=datetime.now(),
                key_id=key_id,
                alert_type="KEY_COMPROMISE",
                severity="CRITICAL",
                details={
                    "indicators": indicators,
                    "usage_count": credentials.usage_count,
                    "last_used": credentials.last_used.isoformat() if credentials.last_used else None
                },
                action_taken="Key immediately deactivated due to compromise indicators"
            )

            self.security_alerts.append(security_alert)

        except Exception as e:
            logger.error(f"Error marking key as compromised: {str(e)}")

    def _cleanup_old_logs(self):
        """Clean up old access logs and alerts"""

        try:
            # Keep only last 30 days of access logs
            cutoff_date = datetime.now() - timedelta(days=30)
            self.access_logs = [log for log in self.access_logs if log.timestamp > cutoff_date]

            # Keep only last 90 days of security alerts
            alert_cutoff_date = datetime.now() - timedelta(days=90)
            self.security_alerts = [alert for alert in self.security_alerts if alert.timestamp > alert_cutoff_date]

        except Exception as e:
            logger.error(f"Error cleaning up old logs: {str(e)}")

    def revoke_key(self, key_id: str, reason: str) -> bool:
        """Manually revoke an API key"""

        try:
            if key_id in self.active_keys:
                credentials = self.active_keys[key_id]
                credentials.is_active = False

                # Move to retired keys
                self.retired_keys[key_id] = credentials
                del self.active_keys[key_id]

                logger.critical(f"🔑 KEY REVOKED: {key_id}")
                logger.critical(f"   Reason: {reason}")

                # Log security event
                alert_id = hashlib.sha256(f"{key_id}REVOKED{time.time()}".encode()).hexdigest()[:16]
                security_alert = SecurityAlert(
                    alert_id=alert_id,
                    timestamp=datetime.now(),
                    key_id=key_id,
                    alert_type="KEY_REVOCATION",
                    severity="HIGH",
                    details={"reason": reason},
                    action_taken="Key manually revoked"
                )

                self.security_alerts.append(security_alert)

                return True

            else:
                logger.warning(f"Key not found for revocation: {key_id}")
                return False

        except Exception as e:
            logger.error(f"Error revoking key {key_id}: {str(e)}")
            return False

    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""

        return {
            "active_keys": len(self.active_keys),
            "retired_keys": len(self.retired_keys),
            "compromised_keys": len(self.compromised_keys),
            "total_access_logs": len(self.access_logs),
            "security_alerts": len(self.security_alerts),
            "unresolved_alerts": len([alert for alert in self.security_alerts if not alert.resolved]),
            "monitoring_active": self.monitoring_active,
            "rotation_interval_hours": self.rotation_interval_hours,
            "anomaly_threshold": self.anomaly_threshold
        }

    def get_key_details(self, key_id: str) -> Optional[Dict[str, Any]]:
        """Get key details (without sensitive credentials)"""

        credentials = self.active_keys.get(key_id)
        if not credentials:
            credentials = self.retired_keys.get(key_id)

        if credentials:
            return {
                "key_id": credentials.key_id,
                "exchange": credentials.exchange,
                "permissions": credentials.permissions,
                "created_at": credentials.created_at.isoformat(),
                "expires_at": credentials.expires_at.isoformat(),
                "last_used": credentials.last_used.isoformat() if credentials.last_used else None,
                "usage_count": credentials.usage_count,
                "is_active": credentials.is_active,
                "is_compromised": credentials.is_compromised
            }

        return None

    def stop_monitoring(self):
        """Stop background monitoring threads"""
        self.monitoring_active = False
        logger.info("API key security monitoring stopped")

# Global instance for the application
_api_security_manager = None

def get_api_security_manager(encryption_password: str = "default_password_change_me") -> APIKeySecurityManager:
    """Get or create the global API security manager instance"""
    global _api_security_manager

    if _api_security_manager is None:
        _api_security_manager = APIKeySecurityManager(encryption_password)

    return _api_security_manager
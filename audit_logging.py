"""
Audit Logging System
Immutable, tamper-proof logging of all critical operations
"""

import json
import hashlib
import logging
from datetime import datetime
from typing import Dict, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of auditable events."""

    # Authentication & Authorization
    LOGIN = "LOGIN"
    LOGOUT = "LOGOUT"
    AUTH_FAILURE = "AUTH_FAILURE"
    API_KEY_CREATED = "API_KEY_CREATED"
    API_KEY_REVOKED = "API_KEY_REVOKED"

    # Trading Operations
    ORDER_PLACED = "ORDER_PLACED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    ORDER_REJECTED = "ORDER_REJECTED"
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_CLOSED = "POSITION_CLOSED"

    # Risk Events
    RISK_LIMIT_BREACH = "RISK_LIMIT_BREACH"
    CIRCUIT_BREAKER_TRIGGERED = "CIRCUIT_BREAKER_TRIGGERED"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    DRAWDOWN_LIMIT_HIT = "DRAWDOWN_LIMIT_HIT"

    # Model Operations
    MODEL_LOADED = "MODEL_LOADED"
    PREDICTION_MADE = "PREDICTION_MADE"
    MODEL_UPDATED = "MODEL_UPDATED"
    MODEL_DEPLOYED = "MODEL_DEPLOYED"

    # Configuration Changes
    CONFIG_CHANGED = "CONFIG_CHANGED"
    LIMITS_CHANGED = "LIMITS_CHANGED"
    CREDENTIALS_ACCESSED = "CREDENTIALS_ACCESSED"

    # Data Operations
    DATA_DOWNLOADED = "DATA_DOWNLOADED"
    DATA_VALIDATED = "DATA_VALIDATED"
    DATA_VALIDATION_FAILED = "DATA_VALIDATION_FAILED"
    DATABASE_WRITE = "DATABASE_WRITE"

    # System Events
    SYSTEM_START = "SYSTEM_START"
    SYSTEM_STOP = "SYSTEM_STOP"
    ERROR_OCCURRED = "ERROR_OCCURRED"


@dataclass
class AuditEvent:
    """Immutable audit event record."""

    timestamp: str
    event_type: AuditEventType
    user: str
    action: str
    resource: str
    status: str  # SUCCESS, FAILURE, WARNING
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    session_id: Optional[str] = None
    previous_hash: Optional[str] = None
    current_hash: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        data = asdict(self)
        data['event_type'] = self.event_type.value
        return data

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True)

    def calculate_hash(self) -> str:
        """
        Calculate SHA-256 hash of event.

        This creates a chain of hashes to detect tampering.
        """
        # Create deterministic representation
        data = self.to_json()
        return hashlib.sha256(data.encode()).hexdigest()


class AuditLogger:
    """
    Immutable audit logger with hash chaining.

    Each event is hashed, and includes the hash of the previous event.
    This creates a blockchain-like chain that detects tampering.
    """

    def __init__(
        self,
        log_file: str = "audit.log",
        log_dir: str = "logs/audit",
        enable_console: bool = False
    ):
        """
        Initialize audit logger.

        Args:
            log_file: Name of audit log file
            log_dir: Directory for audit logs
            enable_console: Also log to console (for debugging)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / log_file
        self.enable_console = enable_console

        # Last event hash for chaining
        self._last_hash: Optional[str] = None

        # Load last hash from file
        self._load_last_hash()

        logger.info(f"Audit logger initialized: {self.log_file}")

    def _load_last_hash(self):
        """Load the last event hash from log file."""
        if self.log_file.exists():
            try:
                # Read last line of log file
                with open(self.log_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        last_line = lines[-1].strip()
                        if last_line:
                            event_data = json.loads(last_line)
                            self._last_hash = event_data.get('current_hash')
                            logger.info(f"Loaded last hash: {self._last_hash[:16]}...")
            except Exception as e:
                logger.error(f"Error loading last hash: {e}")

    def log(
        self,
        event_type: AuditEventType,
        user: str,
        action: str,
        resource: str,
        status: str = "SUCCESS",
        details: Optional[Dict] = None,
        ip_address: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """
        Log an audit event.

        Args:
            event_type: Type of event
            user: User who performed action
            action: Action performed
            resource: Resource affected
            status: SUCCESS, FAILURE, WARNING
            details: Additional details
            ip_address: IP address of user
            session_id: Session identifier
        """
        # Create event
        event = AuditEvent(
            timestamp=datetime.utcnow().isoformat() + 'Z',
            event_type=event_type,
            user=user,
            action=action,
            resource=resource,
            status=status,
            details=details or {},
            ip_address=ip_address,
            session_id=session_id,
            previous_hash=self._last_hash
        )

        # Calculate hash
        event.current_hash = event.calculate_hash()

        # Update last hash
        self._last_hash = event.current_hash

        # Write to log file (append-only)
        try:
            with open(self.log_file, 'a') as f:
                f.write(event.to_json() + '\n')
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
            # This is critical - audit logging failure should be visible
            raise

        # Optional console output
        if self.enable_console:
            print(f"[AUDIT] {event.timestamp} | {event_type.value} | {user} | {action} | {status}")

        logger.debug(f"Audit event logged: {event_type.value} by {user}")

    def verify_chain(self) -> tuple[bool, Optional[str]]:
        """
        Verify the integrity of the audit log chain.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.log_file.exists():
            return True, None

        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()

            previous_hash = None
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue

                event_data = json.loads(line)

                # Check previous hash matches (chain linkage)
                if event_data.get('previous_hash') != previous_hash:
                    return False, f"Hash chain broken at line {i+1}: expected previous_hash={previous_hash}, got={event_data.get('previous_hash')}"

                # Verify current hash exists and is valid format
                current_hash = event_data.get('current_hash')
                if not current_hash or len(current_hash) != 64:
                    return False, f"Invalid or missing hash at line {i+1}"

                # Update previous_hash for next iteration
                previous_hash = current_hash

            return True, None

        except Exception as e:
            return False, f"Verification error: {e}"

    def get_events(
        self,
        event_type: Optional[AuditEventType] = None,
        user: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> list[Dict]:
        """
        Query audit events.

        Args:
            event_type: Filter by event type
            user: Filter by user
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum number of events to return

        Returns:
            List of event dictionaries
        """
        if not self.log_file.exists():
            return []

        events = []

        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    event_data = json.loads(line)

                    # Apply filters
                    if event_type and event_data['event_type'] != event_type.value:
                        continue

                    if user and event_data['user'] != user:
                        continue

                    event_time = datetime.fromisoformat(event_data['timestamp'].rstrip('Z'))

                    if start_time and event_time < start_time:
                        continue

                    if end_time and event_time > end_time:
                        continue

                    events.append(event_data)

                    if len(events) >= limit:
                        break

        except Exception as e:
            logger.error(f"Error querying audit events: {e}")

        return events


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def audit_log(
    event_type: AuditEventType,
    user: str,
    action: str,
    resource: str,
    status: str = "SUCCESS",
    **kwargs
):
    """
    Convenience function to log audit event.

    Args:
        event_type: Type of event
        user: User who performed action
        action: Action description
        resource: Resource affected
        status: SUCCESS, FAILURE, WARNING
        **kwargs: Additional fields (details, ip_address, session_id)
    """
    logger = get_audit_logger()
    logger.log(event_type, user, action, resource, status, **kwargs)


# Decorator for auditing function calls
def audit_function(event_type: AuditEventType, resource_param: str = None):
    """
    Decorator to automatically audit function calls.

    Args:
        event_type: Type of audit event
        resource_param: Name of parameter to use as resource

    Example:
        @audit_function(AuditEventType.ORDER_PLACED, resource_param='symbol')
        def place_order(symbol, quantity, user='system'):
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get user from kwargs or use 'system'
            user = kwargs.get('user', 'system')

            # Get resource
            resource = 'unknown'
            if resource_param and resource_param in kwargs:
                resource = kwargs[resource_param]

            # Call function
            try:
                result = func(*args, **kwargs)

                # Log success
                audit_log(
                    event_type=event_type,
                    user=user,
                    action=f"{func.__name__}",
                    resource=resource,
                    status="SUCCESS",
                    details={'args': str(args), 'kwargs': str(kwargs)}
                )

                return result

            except Exception as e:
                # Log failure
                audit_log(
                    event_type=event_type,
                    user=user,
                    action=f"{func.__name__}",
                    resource=resource,
                    status="FAILURE",
                    details={'error': str(e)}
                )
                raise

        return wrapper
    return decorator

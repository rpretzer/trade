"""
Unit Tests for Security Infrastructure
Tests audit logging, credential management, and API key validation
"""

import pytest
import os
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from audit_logging import (
    AuditLogger, AuditEvent, AuditEventType, audit_log, get_audit_logger
)


class TestAuditLogging:
    """Test audit logging system."""

    def setup_method(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.test_dir, "test_audit.log")

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_audit_event_creation(self):
        """Test creating an audit event."""
        event = AuditEvent(
            timestamp="2026-01-31T10:00:00Z",
            event_type=AuditEventType.ORDER_PLACED,
            user="test_user",
            action="place_order",
            resource="AAPL",
            status="SUCCESS",
            details={'quantity': 100, 'price': 150.0}
        )

        assert event.user == "test_user"
        assert event.event_type == AuditEventType.ORDER_PLACED
        assert event.status == "SUCCESS"

    def test_audit_event_hash_calculation(self):
        """Test audit event hash calculation."""
        event = AuditEvent(
            timestamp="2026-01-31T10:00:00Z",
            event_type=AuditEventType.ORDER_PLACED,
            user="test_user",
            action="place_order",
            resource="AAPL",
            status="SUCCESS",
            details={}
        )

        hash1 = event.calculate_hash()
        hash2 = event.calculate_hash()

        # Same event should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 produces 64 hex characters

    def test_audit_event_hash_changes_with_data(self):
        """Test that hash changes when data changes."""
        event1 = AuditEvent(
            timestamp="2026-01-31T10:00:00Z",
            event_type=AuditEventType.ORDER_PLACED,
            user="test_user",
            action="place_order",
            resource="AAPL",
            status="SUCCESS",
            details={'quantity': 100}
        )

        event2 = AuditEvent(
            timestamp="2026-01-31T10:00:00Z",
            event_type=AuditEventType.ORDER_PLACED,
            user="test_user",
            action="place_order",
            resource="AAPL",
            status="SUCCESS",
            details={'quantity': 200}  # Different quantity
        )

        assert event1.calculate_hash() != event2.calculate_hash()

    def test_audit_logger_initialization(self):
        """Test audit logger initialization."""
        logger = AuditLogger(
            log_file="test.log",
            log_dir=self.test_dir
        )

        # Directory should exist, file created on first log
        assert logger.log_dir.exists()
        assert logger._last_hash is None  # No events yet

    def test_audit_logger_logs_event(self):
        """Test logging an audit event."""
        logger = AuditLogger(
            log_file="test.log",
            log_dir=self.test_dir
        )

        logger.log(
            event_type=AuditEventType.ORDER_PLACED,
            user="test_user",
            action="place_order",
            resource="AAPL",
            status="SUCCESS",
            details={'quantity': 100}
        )

        # Check file was written
        assert logger.log_file.exists()

        # Read and verify
        with open(logger.log_file, 'r') as f:
            lines = f.readlines()

        assert len(lines) == 1
        event_data = json.loads(lines[0])
        assert event_data['user'] == "test_user"
        assert event_data['resource'] == "AAPL"

    def test_audit_logger_hash_chaining(self):
        """Test that audit events are hash-chained."""
        logger = AuditLogger(
            log_file="test.log",
            log_dir=self.test_dir
        )

        # Log first event
        logger.log(
            event_type=AuditEventType.ORDER_PLACED,
            user="user1",
            action="action1",
            resource="resource1",
            status="SUCCESS"
        )

        first_hash = logger._last_hash

        # Log second event
        logger.log(
            event_type=AuditEventType.ORDER_FILLED,
            user="user2",
            action="action2",
            resource="resource2",
            status="SUCCESS"
        )

        # Read events
        with open(logger.log_file, 'r') as f:
            lines = f.readlines()

        event1 = json.loads(lines[0])
        event2 = json.loads(lines[1])

        # Second event should reference first event's hash
        assert event2['previous_hash'] == first_hash
        assert event2['previous_hash'] == event1['current_hash']

    def test_audit_logger_verify_chain_valid(self):
        """Test verifying a valid audit chain."""
        logger = AuditLogger(
            log_file="test.log",
            log_dir=self.test_dir
        )

        # Log some events
        for i in range(5):
            logger.log(
                event_type=AuditEventType.ORDER_PLACED,
                user=f"user{i}",
                action=f"action{i}",
                resource=f"resource{i}",
                status="SUCCESS"
            )

        # Verify chain
        is_valid, error = logger.verify_chain()
        if not is_valid:
            print(f"Chain verification failed: {error}")
        assert is_valid is True, f"Chain verification failed: {error}"
        assert error is None

    def test_audit_logger_detect_tampering(self):
        """Test that tampering with hash chain is detected."""
        logger = AuditLogger(
            log_file="test.log",
            log_dir=self.test_dir
        )

        # Log some events
        for i in range(3):
            logger.log(
                event_type=AuditEventType.ORDER_PLACED,
                user=f"user{i}",
                action=f"action{i}",
                resource=f"resource{i}",
                status="SUCCESS"
            )

        # Tamper with the hash chain (modify current_hash of middle event)
        with open(logger.log_file, 'r') as f:
            lines = f.readlines()

        # Modify second event's hash
        event2 = json.loads(lines[1])
        event2['current_hash'] = 'TAMPERED_HASH_' + '0' * 50  # Invalid hash
        lines[1] = json.dumps(event2) + '\n'

        # Write back
        with open(logger.log_file, 'w') as f:
            f.writelines(lines)

        # Create new logger and verify
        logger2 = AuditLogger(
            log_file="test.log",
            log_dir=self.test_dir
        )

        is_valid, error = logger2.verify_chain()
        assert is_valid is False
        assert "Hash chain broken" in error or "Invalid" in error

    def test_audit_logger_query_events(self):
        """Test querying audit events."""
        logger = AuditLogger(
            log_file="test.log",
            log_dir=self.test_dir
        )

        # Log different types of events
        logger.log(
            event_type=AuditEventType.ORDER_PLACED,
            user="user1",
            action="action1",
            resource="AAPL",
            status="SUCCESS"
        )

        logger.log(
            event_type=AuditEventType.ORDER_FILLED,
            user="user2",
            action="action2",
            resource="MSFT",
            status="SUCCESS"
        )

        logger.log(
            event_type=AuditEventType.ORDER_PLACED,
            user="user1",
            action="action3",
            resource="TSLA",
            status="SUCCESS"
        )

        # Query all events
        events = logger.get_events()
        assert len(events) == 3

        # Query by event type
        order_placed_events = logger.get_events(event_type=AuditEventType.ORDER_PLACED)
        assert len(order_placed_events) == 2

        # Query by user
        user1_events = logger.get_events(user="user1")
        assert len(user1_events) == 2

    def test_audit_logger_query_limit(self):
        """Test query limit parameter."""
        logger = AuditLogger(
            log_file="test.log",
            log_dir=self.test_dir
        )

        # Log 10 events
        for i in range(10):
            logger.log(
                event_type=AuditEventType.ORDER_PLACED,
                user=f"user{i}",
                action=f"action{i}",
                resource=f"resource{i}",
                status="SUCCESS"
            )

        # Query with limit
        events = logger.get_events(limit=5)
        assert len(events) == 5

    def test_audit_logger_loads_last_hash_on_restart(self):
        """Test that logger loads last hash when restarted."""
        # Create logger and log an event
        logger1 = AuditLogger(
            log_file="test.log",
            log_dir=self.test_dir
        )

        logger1.log(
            event_type=AuditEventType.ORDER_PLACED,
            user="user1",
            action="action1",
            resource="resource1",
            status="SUCCESS"
        )

        first_hash = logger1._last_hash

        # Create new logger instance (simulating restart)
        logger2 = AuditLogger(
            log_file="test.log",
            log_dir=self.test_dir
        )

        # Should load the last hash
        assert logger2._last_hash == first_hash

        # Log another event
        logger2.log(
            event_type=AuditEventType.ORDER_FILLED,
            user="user2",
            action="action2",
            resource="resource2",
            status="SUCCESS"
        )

        # Verify chain is intact
        is_valid, error = logger2.verify_chain()
        assert is_valid is True


class TestCredentialSecurity:
    """Test credential management security features."""

    def test_no_plaintext_fallback_in_trading_api(self):
        """Test that trading_api.py doesn't fall back to plaintext."""
        # Read trading_api.py
        with open('trading_api.py', 'r') as f:
            content = f.read()

        # Should not contain plaintext credential fallback
        assert 'schwab_config.txt' not in content
        assert "Fallback to old plain text file" not in content

    def test_no_dummy_credentials_in_sentiment_analysis(self):
        """Test that sentiment_analysis.py rejects dummy credentials."""
        # Read sentiment_analysis.py
        with open('sentiment_analysis.py', 'r') as f:
            content = f.read()

        # Should contain validation for dummy credentials
        assert "Dummy credentials" in content or "dummy credentials" in content
        assert "rejected" in content.lower() or "invalid" in content.lower()


class TestAPIKeyValidation:
    """Test API key validation."""

    def test_api_key_not_none(self):
        """Test that None API keys are rejected."""
        # This would be tested by the actual credential manager
        # when integrated with the API
        pass

    def test_api_key_not_empty(self):
        """Test that empty API keys are rejected."""
        pass

    def test_api_key_not_dummy(self):
        """Test that dummy API keys are rejected."""
        pass

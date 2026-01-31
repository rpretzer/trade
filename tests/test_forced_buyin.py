"""
Unit Tests for Forced Buy-In Functionality
Tests forced buy-in scenarios for short positions
"""

import pytest
from datetime import datetime, timedelta
from trading_execution import (
    ForcedBuyInManager, ForcedBuyInEvent, ForcedBuyInReason
)


class TestForcedBuyInManager:
    """Test forced buy-in management."""

    def setup_method(self):
        """Set up test environment."""
        self.manager = ForcedBuyInManager(
            borrow_rate_threshold=0.20,
            dividend_lookback_days=3
        )
        self.test_time = datetime(2024, 1, 15, 10, 0)

    def test_initialization(self):
        """Test manager initialization."""
        assert self.manager.borrow_rate_threshold == 0.20
        assert self.manager.dividend_lookback_days == 3
        assert len(self.manager.active_events) == 0

    def test_borrow_rate_increase_triggers_buyin(self):
        """Test that significant borrow rate increase triggers buy-in."""
        event = self.manager.check_borrow_rate_increase(
            symbol='TSLA',
            old_rate=0.05,  # 5%
            new_rate=0.25,  # 25% (5x increase)
            quantity=100,
            timestamp=self.test_time
        )

        assert event is not None
        assert event.symbol == 'TSLA'
        assert event.quantity == 100
        assert event.reason == ForcedBuyInReason.HARD_TO_BORROW
        assert event.penalty_rate == 0.25
        assert len(self.manager.active_events) == 1

    def test_borrow_rate_small_increase_no_trigger(self):
        """Test that small rate increase doesn't trigger buy-in."""
        event = self.manager.check_borrow_rate_increase(
            symbol='AAPL',
            old_rate=0.05,
            new_rate=0.07,  # Only 40% increase
            quantity=100,
            timestamp=self.test_time
        )

        assert event is None
        assert len(self.manager.active_events) == 0

    def test_borrow_rate_below_threshold_no_trigger(self):
        """Test that rate below threshold doesn't trigger."""
        event = self.manager.check_borrow_rate_increase(
            symbol='AAPL',
            old_rate=0.05,
            new_rate=0.15,  # 3x increase but below 20% threshold
            quantity=100,
            timestamp=self.test_time
        )

        assert event is None

    def test_broker_recall(self):
        """Test broker share recall."""
        event = self.manager.check_broker_recall(
            symbol='GME',
            quantity=200,
            timestamp=self.test_time,
            notice_hours=24
        )

        assert event is not None
        assert event.symbol == 'GME'
        assert event.quantity == 200
        assert event.reason == ForcedBuyInReason.BROKER_RECALL
        assert event.deadline == self.test_time + timedelta(hours=24)
        assert event.penalty_rate == 0.10
        assert len(self.manager.active_events) == 1

    def test_broker_recall_short_notice(self):
        """Test broker recall with short notice."""
        event = self.manager.check_broker_recall(
            symbol='AMC',
            quantity=500,
            timestamp=self.test_time,
            notice_hours=4  # Only 4 hours
        )

        assert event is not None
        assert event.deadline == self.test_time + timedelta(hours=4)

    def test_dividend_approaching_triggers_buyin(self):
        """Test that approaching dividend triggers buy-in."""
        ex_dividend_date = self.test_time + timedelta(days=2)

        event = self.manager.check_dividend_approaching(
            symbol='AAPL',
            quantity=100,
            ex_dividend_date=ex_dividend_date,
            current_time=self.test_time,
            dividend_amount=0.50,  # $0.50 dividend
            price=20.0  # 2.5% yield
        )

        assert event is not None
        assert event.symbol == 'AAPL'
        assert event.reason == ForcedBuyInReason.DIVIDEND_APPROACHING
        assert event.deadline == ex_dividend_date - timedelta(days=1)
        assert len(self.manager.active_events) == 1

    def test_dividend_too_small_no_trigger(self):
        """Test that small dividend doesn't trigger buy-in."""
        ex_dividend_date = self.test_time + timedelta(days=2)

        event = self.manager.check_dividend_approaching(
            symbol='MSFT',
            quantity=100,
            ex_dividend_date=ex_dividend_date,
            current_time=self.test_time,
            dividend_amount=0.10,  # $0.10 dividend
            price=300.0  # Only 0.03% yield
        )

        assert event is None  # Dividend too small

    def test_dividend_too_far_no_trigger(self):
        """Test that distant dividend doesn't trigger buy-in."""
        ex_dividend_date = self.test_time + timedelta(days=10)  # 10 days away

        event = self.manager.check_dividend_approaching(
            symbol='MSFT',
            quantity=100,
            ex_dividend_date=ex_dividend_date,
            current_time=self.test_time,
            dividend_amount=0.50,
            price=20.0
        )

        assert event is None  # Too far away

    def test_corporate_action(self):
        """Test corporate action triggers buy-in."""
        action_date = self.test_time + timedelta(days=5)

        event = self.manager.check_corporate_action(
            symbol='ABC',
            quantity=300,
            action_type='MERGER',
            action_date=action_date,
            current_time=self.test_time
        )

        assert event is not None
        assert event.symbol == 'ABC'
        assert event.quantity == 300
        assert event.reason == ForcedBuyInReason.CORPORATE_ACTION
        assert event.deadline == action_date - timedelta(days=1)
        assert event.penalty_rate == 0.20  # High penalty
        assert len(self.manager.active_events) == 1

    def test_get_active_events_all(self):
        """Test getting all active events."""
        # Create multiple events
        self.manager.check_broker_recall('AAPL', 100, self.test_time)
        self.manager.check_broker_recall('MSFT', 200, self.test_time)

        events = self.manager.get_active_events()
        assert len(events) == 2

    def test_get_active_events_by_symbol(self):
        """Test filtering events by symbol."""
        self.manager.check_broker_recall('AAPL', 100, self.test_time)
        self.manager.check_broker_recall('MSFT', 200, self.test_time)

        aapl_events = self.manager.get_active_events(symbol='AAPL')
        assert len(aapl_events) == 1
        assert aapl_events[0].symbol == 'AAPL'

    def test_resolve_event_on_time(self):
        """Test resolving event before deadline."""
        event = self.manager.check_broker_recall('AAPL', 100, self.test_time)

        # Close position before deadline
        close_time = self.test_time + timedelta(hours=12)
        result = self.manager.resolve_event(event, close_time)

        assert result['symbol'] == 'AAPL'
        assert result['closed_on_time'] is True
        assert result['penalty_rate'] == 0.0
        assert len(self.manager.active_events) == 0

    def test_resolve_event_late(self):
        """Test resolving event after deadline (penalty applied)."""
        event = self.manager.check_broker_recall('AAPL', 100, self.test_time)

        # Close position after deadline
        close_time = self.test_time + timedelta(hours=30)  # Missed 24hr deadline
        result = self.manager.resolve_event(event, close_time)

        assert result['symbol'] == 'AAPL'
        assert result['closed_on_time'] is False
        assert result['penalty_rate'] == 0.10  # Penalty applied
        assert len(self.manager.active_events) == 0

    def test_multiple_events_same_symbol(self):
        """Test multiple events for same symbol."""
        # Broker recall
        event1 = self.manager.check_broker_recall('TSLA', 100, self.test_time)

        # Dividend approaching
        ex_dividend = self.test_time + timedelta(days=2)
        event2 = self.manager.check_dividend_approaching(
            'TSLA', 100, ex_dividend, self.test_time,
            dividend_amount=1.0, price=20.0
        )

        assert len(self.manager.active_events) == 2
        tsla_events = self.manager.get_active_events(symbol='TSLA')
        assert len(tsla_events) == 2


class TestForcedBuyInEvent:
    """Test ForcedBuyInEvent dataclass."""

    def test_event_creation(self):
        """Test creating a forced buy-in event."""
        timestamp = datetime(2024, 1, 15, 10, 0)
        deadline = timestamp + timedelta(days=1)

        event = ForcedBuyInEvent(
            symbol='AAPL',
            quantity=100,
            reason=ForcedBuyInReason.BROKER_RECALL,
            timestamp=timestamp,
            deadline=deadline,
            penalty_rate=0.05
        )

        assert event.symbol == 'AAPL'
        assert event.quantity == 100
        assert event.reason == ForcedBuyInReason.BROKER_RECALL
        assert event.penalty_rate == 0.05

    def test_event_to_dict(self):
        """Test converting event to dictionary."""
        timestamp = datetime(2024, 1, 15, 10, 0)
        deadline = timestamp + timedelta(days=1)

        event = ForcedBuyInEvent(
            symbol='MSFT',
            quantity=200,
            reason=ForcedBuyInReason.HARD_TO_BORROW,
            timestamp=timestamp,
            deadline=deadline
        )

        data = event.to_dict()
        assert data['symbol'] == 'MSFT'
        assert data['quantity'] == 200
        assert data['reason'] == 'HARD_TO_BORROW'
        assert 'timestamp' in data
        assert 'deadline' in data
        assert data['penalty_rate'] == 0.05  # Default


class TestForcedBuyInReason:
    """Test ForcedBuyInReason enum."""

    def test_all_reasons_defined(self):
        """Test that all expected reasons are defined."""
        assert ForcedBuyInReason.BROKER_RECALL
        assert ForcedBuyInReason.HARD_TO_BORROW
        assert ForcedBuyInReason.DIVIDEND_APPROACHING
        assert ForcedBuyInReason.CORPORATE_ACTION
        assert ForcedBuyInReason.POSITION_LIMIT

    def test_reason_values(self):
        """Test enum values are correct."""
        assert ForcedBuyInReason.BROKER_RECALL.value == 'BROKER_RECALL'
        assert ForcedBuyInReason.HARD_TO_BORROW.value == 'HARD_TO_BORROW'

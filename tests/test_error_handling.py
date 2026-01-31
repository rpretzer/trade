"""
Unit Tests for Error Handling Infrastructure
Tests retry logic, circuit breakers, rate limiters, and dead man's switch
"""

import pytest
import time
from datetime import datetime
from unittest.mock import Mock, patch
from exceptions import (
    APIException, APIConnectionError, CircuitBreakerOpenError,
    CircuitBreakerTriggeredError
)
from error_handling import (
    retry, RetryConfig, CircuitBreaker, DeadMansSwitch, RateLimiter
)


class TestRetryDecorator:
    """Test retry decorator with exponential backoff."""

    def test_retry_succeeds_first_attempt(self):
        """Test that successful function doesn't retry."""
        call_count = [0]

        @retry(exceptions=(APIException,))
        def succeeds_immediately():
            call_count[0] += 1
            return "success"

        result = succeeds_immediately()
        assert result == "success"
        assert call_count[0] == 1

    def test_retry_succeeds_after_failures(self):
        """Test that function retries and eventually succeeds."""
        call_count = [0]

        @retry(exceptions=(APIException,), config=RetryConfig(max_attempts=3, initial_delay=0.1))
        def succeeds_on_third_try():
            call_count[0] += 1
            if call_count[0] < 3:
                raise APIException("Temporary failure")
            return "success"

        result = succeeds_on_third_try()
        assert result == "success"
        assert call_count[0] == 3

    def test_retry_fails_after_max_attempts(self):
        """Test that function raises exception after max attempts."""
        call_count = [0]

        @retry(exceptions=(APIException,), config=RetryConfig(max_attempts=3, initial_delay=0.1))
        def always_fails():
            call_count[0] += 1
            raise APIException("Permanent failure")

        with pytest.raises(APIException, match="Permanent failure"):
            always_fails()

        assert call_count[0] == 3

    def test_retry_doesnt_catch_wrong_exception(self):
        """Test that retry doesn't catch exceptions not in the list."""
        call_count = [0]

        @retry(exceptions=(APIException,), config=RetryConfig(max_attempts=3))
        def raises_wrong_exception():
            call_count[0] += 1
            raise ValueError("Different exception")

        with pytest.raises(ValueError):
            raises_wrong_exception()

        assert call_count[0] == 1  # Should not retry

    def test_retry_exponential_backoff(self):
        """Test that retry uses exponential backoff."""
        call_times = []

        @retry(
            exceptions=(APIException,),
            config=RetryConfig(max_attempts=4, initial_delay=0.1, exponential_base=2.0, jitter=False)
        )
        def track_retry_times():
            call_times.append(time.time())
            if len(call_times) < 4:
                raise APIException("Retry")
            return "success"

        track_retry_times()

        # Check delays between calls
        delays = [call_times[i+1] - call_times[i] for i in range(len(call_times)-1)]

        # Delays should be approximately: 0.1s, 0.2s, 0.4s (exponential)
        assert delays[0] == pytest.approx(0.1, abs=0.05)
        assert delays[1] == pytest.approx(0.2, abs=0.05)
        assert delays[2] == pytest.approx(0.4, abs=0.05)

    def test_retry_with_callback(self):
        """Test that on_retry callback is called."""
        callback_calls = []

        def on_retry(attempt, exception, wait_time):
            callback_calls.append((attempt, str(exception), wait_time))

        @retry(
            exceptions=(APIException,),
            config=RetryConfig(max_attempts=3, initial_delay=0.1, jitter=False),
            on_retry=on_retry
        )
        def fails_twice():
            if len(callback_calls) < 2:
                raise APIException("Retry me")
            return "success"

        fails_twice()

        assert len(callback_calls) == 2
        assert callback_calls[0][0] == 1  # First retry
        assert callback_calls[1][0] == 2  # Second retry


class TestCircuitBreaker:
    """Test circuit breaker pattern."""

    def test_circuit_closed_passes_calls(self):
        """Test that calls pass through when circuit is closed."""
        cb = CircuitBreaker(name="test", failure_threshold=3)

        result = cb.call(lambda: "success")
        assert result == "success"
        assert cb.state == CircuitBreaker.STATE_CLOSED

    def test_circuit_opens_after_failures(self):
        """Test that circuit opens after threshold failures."""
        cb = CircuitBreaker(name="test", failure_threshold=3)

        # Cause 3 failures
        for i in range(3):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        # Circuit should now be open
        assert cb.state == CircuitBreaker.STATE_OPEN

    def test_circuit_open_blocks_calls(self):
        """Test that circuit breaker blocks calls when open."""
        cb = CircuitBreaker(name="test", failure_threshold=2)

        # Cause 2 failures to open circuit
        for i in range(2):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        # Next call should be blocked
        with pytest.raises(CircuitBreakerOpenError):
            cb.call(lambda: "should be blocked")

    def test_circuit_half_open_after_timeout(self):
        """Test that circuit transitions to half-open after timeout."""
        cb = CircuitBreaker(
            name="test",
            failure_threshold=2,
            recovery_timeout=0.5  # 0.5 second timeout
        )

        # Open the circuit
        for i in range(2):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert cb.state == CircuitBreaker.STATE_OPEN

        # Wait for recovery timeout
        time.sleep(0.6)

        # Check state (should transition to HALF_OPEN when checked)
        assert cb.state == CircuitBreaker.STATE_HALF_OPEN

    def test_circuit_closes_after_successes_in_half_open(self):
        """Test that circuit closes after enough successes in half-open."""
        cb = CircuitBreaker(
            name="test",
            failure_threshold=2,
            recovery_timeout=0.2,
            success_threshold=2
        )

        # Open the circuit
        for i in range(2):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        # Wait for recovery
        time.sleep(0.3)

        # Make successful calls in HALF_OPEN state
        cb.call(lambda: "success1")
        assert cb.state == CircuitBreaker.STATE_HALF_OPEN

        cb.call(lambda: "success2")
        assert cb.state == CircuitBreaker.STATE_CLOSED

    def test_circuit_reopens_on_failure_in_half_open(self):
        """Test that circuit reopens on failure in half-open state."""
        cb = CircuitBreaker(
            name="test",
            failure_threshold=2,
            recovery_timeout=0.2
        )

        # Open the circuit
        for i in range(2):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        # Wait for recovery
        time.sleep(0.3)
        assert cb.state == CircuitBreaker.STATE_HALF_OPEN

        # Fail in HALF_OPEN state
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("fail again")))

        # Should be back to OPEN
        assert cb.state == CircuitBreaker.STATE_OPEN

    def test_circuit_breaker_callbacks(self):
        """Test that callbacks are called on state transitions."""
        opened = [False]
        closed = [False]

        def on_open(cb):
            opened[0] = True

        def on_close(cb):
            closed[0] = True

        cb = CircuitBreaker(
            name="test",
            failure_threshold=2,
            recovery_timeout=0.2,
            success_threshold=1,
            on_open=on_open,
            on_close=on_close
        )

        # Open circuit
        for i in range(2):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert opened[0] is True

        # Wait and close circuit
        time.sleep(0.3)
        cb.call(lambda: "success")

        assert closed[0] is True

    def test_circuit_manual_reset(self):
        """Test that circuit can be manually reset."""
        cb = CircuitBreaker(name="test", failure_threshold=2)

        # Open circuit
        for i in range(2):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert cb.state == CircuitBreaker.STATE_OPEN

        # Manual reset
        cb.reset()
        assert cb.state == CircuitBreaker.STATE_CLOSED


class TestDeadMansSwitch:
    """Test dead man's switch functionality."""

    def test_heartbeat_resets_timer(self):
        """Test that heartbeat resets the timer."""
        dms = DeadMansSwitch(name="test", timeout=1.0)

        time.sleep(0.5)
        dms.heartbeat()
        time.sleep(0.5)

        # Should still be alive (total 1s but heartbeat at 0.5s)
        assert dms.check() is True

    def test_timeout_triggers_callback(self):
        """Test that timeout triggers callback."""
        triggered = [False]

        def on_timeout(dms):
            triggered[0] = True

        dms = DeadMansSwitch(
            name="test",
            timeout=0.3,
            on_timeout=on_timeout
        )

        # Wait for timeout
        time.sleep(0.4)

        # Check should trigger callback
        result = dms.check()
        assert result is False
        assert triggered[0] is True

    def test_deactivate_prevents_timeout(self):
        """Test that deactivating prevents timeout."""
        triggered = [False]

        def on_timeout(dms):
            triggered[0] = True

        dms = DeadMansSwitch(
            name="test",
            timeout=0.2,
            on_timeout=on_timeout
        )

        dms.deactivate()
        time.sleep(0.3)

        # Should not timeout when deactivated
        assert dms.check() is True
        assert triggered[0] is False


class TestRateLimiter:
    """Test token bucket rate limiter."""

    def test_rate_limiter_allows_within_limit(self):
        """Test that rate limiter allows requests within limit."""
        rl = RateLimiter(name="test", rate=10.0, capacity=10)

        # Should allow 10 requests immediately
        for i in range(10):
            assert rl.acquire() is True

    def test_rate_limiter_blocks_over_limit(self):
        """Test that rate limiter blocks requests over limit."""
        rl = RateLimiter(name="test", rate=10.0, capacity=5)

        # Acquire all tokens
        for i in range(5):
            assert rl.acquire() is True

        # Next request should be blocked
        assert rl.acquire() is False

    def test_rate_limiter_refills_tokens(self):
        """Test that rate limiter refills tokens over time."""
        rl = RateLimiter(name="test", rate=10.0, capacity=5)

        # Acquire all tokens
        for i in range(5):
            rl.acquire()

        # Wait for refill (0.5s should add 5 tokens at rate=10/s)
        time.sleep(0.5)

        # Should be able to acquire again
        assert rl.acquire() is True

    def test_rate_limiter_wait_and_acquire(self):
        """Test that wait_and_acquire waits for tokens."""
        rl = RateLimiter(name="test", rate=10.0, capacity=2)

        # Acquire all tokens
        rl.acquire(2)

        # Wait and acquire should succeed after brief wait
        start_time = time.time()
        rl.wait_and_acquire(tokens=1, timeout=1.0)
        elapsed = time.time() - start_time

        # Should have waited ~0.1s for 1 token at rate=10/s
        assert elapsed == pytest.approx(0.1, abs=0.05)

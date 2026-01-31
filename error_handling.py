"""
Error Handling Infrastructure
Provides retry logic, circuit breakers, and dead man's switch
"""

import time
import logging
from functools import wraps
from typing import Callable, Optional, Tuple, Type
from datetime import datetime, timedelta
from exceptions import (
    APIException, APIRateLimitError, CircuitBreakerOpenError,
    CircuitBreakerTriggeredError
)

logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        """
        Initialize retry configuration.

        Args:
            max_attempts: Maximum number of retry attempts
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential backoff
            jitter: Whether to add random jitter to delays
        """
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter


def retry(
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable] = None
):
    """
    Retry decorator with exponential backoff.

    Args:
        exceptions: Tuple of exception types to retry on
        config: RetryConfig instance (uses default if None)
        on_retry: Optional callback function called on each retry

    Example:
        @retry(exceptions=(APIException,), config=RetryConfig(max_attempts=5))
        def fetch_data():
            # Code that might fail
            pass
    """
    if config is None:
        config = RetryConfig()

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            delay = config.initial_delay

            while attempt < config.max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1

                    if attempt >= config.max_attempts:
                        logger.error(
                            f"Function {func.__name__} failed after {attempt} attempts: {e}"
                        )
                        raise

                    # Calculate delay with exponential backoff
                    if config.jitter:
                        import random
                        jitter_delay = delay * (0.5 + random.random() * 0.5)
                    else:
                        jitter_delay = delay

                    wait_time = min(jitter_delay, config.max_delay)

                    logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt}/{config.max_attempts}): {e}. "
                        f"Retrying in {wait_time:.2f}s..."
                    )

                    if on_retry:
                        on_retry(attempt, e, wait_time)

                    time.sleep(wait_time)
                    delay *= config.exponential_base

            return None

        return wrapper
    return decorator


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.

    States:
        CLOSED: Normal operation, requests pass through
        OPEN: Too many failures, requests blocked
        HALF_OPEN: Testing if service recovered
    """

    STATE_CLOSED = 'CLOSED'
    STATE_OPEN = 'OPEN'
    STATE_HALF_OPEN = 'HALF_OPEN'

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2,
        on_open: Optional[Callable] = None,
        on_close: Optional[Callable] = None
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Name of the circuit breaker
            failure_threshold: Number of failures before opening
            recovery_timeout: Seconds before attempting recovery
            success_threshold: Successes needed in HALF_OPEN to close
            on_open: Callback when circuit opens
            on_close: Callback when circuit closes
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.on_open = on_open
        self.on_close = on_close

        self._state = self.STATE_CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._opened_at = None

    @property
    def state(self) -> str:
        """Get current circuit breaker state."""
        if self._state == self.STATE_OPEN:
            # Check if recovery timeout has elapsed
            if self._opened_at and \
               (datetime.now() - self._opened_at).total_seconds() >= self.recovery_timeout:
                self._transition_to_half_open()

        return self._state

    def call(self, func: Callable, *args, **kwargs):
        """
        Call a function through the circuit breaker.

        Args:
            func: Function to call
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Function return value

        Raises:
            CircuitBreakerOpenError: If circuit is open
        """
        if self.state == self.STATE_OPEN:
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is OPEN. Service unavailable."
            )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        """Handle successful call."""
        if self._state == self.STATE_HALF_OPEN:
            self._success_count += 1
            logger.info(
                f"Circuit breaker '{self.name}': Success in HALF_OPEN "
                f"({self._success_count}/{self.success_threshold})"
            )

            if self._success_count >= self.success_threshold:
                self._transition_to_closed()
        elif self._state == self.STATE_CLOSED:
            # Reset failure count on success
            self._failure_count = 0

    def _on_failure(self):
        """Handle failed call."""
        self._failure_count += 1
        self._last_failure_time = datetime.now()

        logger.warning(
            f"Circuit breaker '{self.name}': Failure "
            f"({self._failure_count}/{self.failure_threshold})"
        )

        if self._state == self.STATE_HALF_OPEN:
            # Failure in HALF_OPEN -> back to OPEN
            self._transition_to_open()
        elif self._state == self.STATE_CLOSED:
            if self._failure_count >= self.failure_threshold:
                self._transition_to_open()

    def _transition_to_open(self):
        """Transition to OPEN state."""
        self._state = self.STATE_OPEN
        self._opened_at = datetime.now()
        self._success_count = 0

        logger.error(
            f"Circuit breaker '{self.name}' opened after "
            f"{self._failure_count} failures"
        )

        if self.on_open:
            self.on_open(self)

    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state."""
        self._state = self.STATE_HALF_OPEN
        self._failure_count = 0
        self._success_count = 0

        logger.info(f"Circuit breaker '{self.name}' entering HALF_OPEN state")

    def _transition_to_closed(self):
        """Transition to CLOSED state."""
        self._state = self.STATE_CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._opened_at = None

        logger.info(f"Circuit breaker '{self.name}' closed")

        if self.on_close:
            self.on_close(self)

    def reset(self):
        """Manually reset circuit breaker to CLOSED state."""
        logger.info(f"Circuit breaker '{self.name}' manually reset")
        self._transition_to_closed()


class DeadMansSwitch:
    """
    Dead man's switch for detecting system failures.

    Requires periodic heartbeat signals. If heartbeat stops,
    triggers emergency shutdown.
    """

    def __init__(
        self,
        name: str,
        timeout: float = 60.0,
        on_timeout: Optional[Callable] = None
    ):
        """
        Initialize dead man's switch.

        Args:
            name: Name of the switch
            timeout: Timeout in seconds without heartbeat
            on_timeout: Callback when timeout occurs
        """
        self.name = name
        self.timeout = timeout
        self.on_timeout = on_timeout
        self._last_heartbeat = datetime.now()
        self._active = True

    def heartbeat(self):
        """Signal that system is alive."""
        self._last_heartbeat = datetime.now()
        logger.debug(f"Dead man's switch '{self.name}': Heartbeat received")

    def check(self) -> bool:
        """
        Check if switch has timed out.

        Returns:
            True if system is alive, False if timed out
        """
        if not self._active:
            return True

        elapsed = (datetime.now() - self._last_heartbeat).total_seconds()

        if elapsed > self.timeout:
            logger.critical(
                f"Dead man's switch '{self.name}' TIMEOUT! "
                f"No heartbeat for {elapsed:.1f}s (limit: {self.timeout}s)"
            )

            if self.on_timeout:
                self.on_timeout(self)

            return False

        return True

    def deactivate(self):
        """Deactivate the switch (for controlled shutdown)."""
        self._active = False
        logger.info(f"Dead man's switch '{self.name}' deactivated")

    def activate(self):
        """Activate the switch."""
        self._active = True
        self._last_heartbeat = datetime.now()
        logger.info(f"Dead man's switch '{self.name}' activated")


class RateLimiter:
    """
    Token bucket rate limiter for API calls.
    """

    def __init__(
        self,
        name: str,
        rate: float,
        capacity: int
    ):
        """
        Initialize rate limiter.

        Args:
            name: Name of the rate limiter
            rate: Token refill rate (tokens per second)
            capacity: Maximum token capacity
        """
        self.name = name
        self.rate = rate
        self.capacity = capacity
        self._tokens = capacity
        self._last_update = time.time()

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_update

        # Add tokens based on elapsed time
        self._tokens = min(
            self.capacity,
            self._tokens + elapsed * self.rate
        )
        self._last_update = now

    def acquire(self, tokens: int = 1) -> bool:
        """
        Attempt to acquire tokens.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens acquired, False if rate limited
        """
        self._refill()

        if self._tokens >= tokens:
            self._tokens -= tokens
            return True

        logger.warning(
            f"Rate limiter '{self.name}': Rate limit exceeded "
            f"({self._tokens:.1f}/{self.capacity} tokens available)"
        )
        return False

    def wait_and_acquire(self, tokens: int = 1, timeout: Optional[float] = None):
        """
        Wait until tokens are available and acquire them.

        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait in seconds

        Raises:
            APIRateLimitError: If timeout is exceeded
        """
        start_time = time.time()

        while True:
            if self.acquire(tokens):
                return

            if timeout and (time.time() - start_time) > timeout:
                raise APIRateLimitError(
                    f"Rate limiter '{self.name}': Timeout waiting for tokens"
                )

            # Wait a bit before trying again
            time.sleep(0.1)

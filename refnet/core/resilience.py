"""
resilience utilities - retry, fallback, circuit breaker patterns.
makes refnet robust against transient failures.
"""

import time
import logging
import functools
from typing import TypeVar, Callable, Optional, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


# setup logging
logger = logging.getLogger("refnet")


T = TypeVar("T")


class CircuitState(Enum):
    """circuit breaker states."""
    CLOSED = "closed"      # normal operation
    OPEN = "open"          # failing, reject calls
    HALF_OPEN = "half_open"  # testing recovery


@dataclass
class RetryConfig:
    """configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True
    # exceptions that should trigger retry
    retryable_exceptions: Tuple[type, ...] = (
        ConnectionError,
        TimeoutError,
        OSError,
    )


@dataclass
class CircuitBreakerConfig:
    """configuration for circuit breaker."""
    failure_threshold: int = 5       # failures before opening
    recovery_timeout: float = 60.0   # seconds before half-open
    half_open_max_calls: int = 3     # test calls in half-open


@dataclass
class CircuitBreaker:
    """
    circuit breaker pattern implementation.
    prevents hammering a failing service.
    """
    name: str
    config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)

    # state
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    half_open_calls: int = 0

    def can_execute(self) -> bool:
        """check if we can make a call."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # check if recovery timeout has passed
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                if elapsed >= self.config.recovery_timeout:
                    logger.info(f"[circuit:{self.name}] transitioning to half-open")
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                    return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            return self.half_open_calls < self.config.half_open_max_calls

        return False

    def record_success(self):
        """record successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_calls += 1
            if self.half_open_calls >= self.config.half_open_max_calls:
                logger.info(f"[circuit:{self.name}] recovery confirmed, closing")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.half_open_calls = 0
        elif self.state == CircuitState.CLOSED:
            # reset failure count on success
            self.failure_count = 0

    def record_failure(self, error: Exception):
        """record failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.state == CircuitState.HALF_OPEN:
            logger.warning(f"[circuit:{self.name}] failure in half-open, reopening")
            self.state = CircuitState.OPEN
            self.half_open_calls = 0
        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                logger.warning(f"[circuit:{self.name}] threshold reached, opening circuit")
                self.state = CircuitState.OPEN

    def reset(self):
        """manually reset circuit breaker."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.half_open_calls = 0
        self.last_failure_time = None


def retry_with_backoff(
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    decorator for retry with exponential backoff.

    usage:
        @retry_with_backoff(RetryConfig(max_attempts=3))
        def flaky_function():
            ...
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(1, config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)

                except config.retryable_exceptions as e:
                    last_exception = e

                    if attempt == config.max_attempts:
                        logger.warning(
                            f"[retry] {func.__name__} failed after {attempt} attempts: {e}"
                        )
                        raise

                    # compute delay with exponential backoff
                    delay = min(
                        config.base_delay * (config.exponential_base ** (attempt - 1)),
                        config.max_delay
                    )

                    # add jitter
                    if config.jitter:
                        import random
                        delay = delay * (0.5 + random.random())

                    logger.info(
                        f"[retry] {func.__name__} attempt {attempt} failed, "
                        f"retrying in {delay:.1f}s: {e}"
                    )

                    if on_retry:
                        on_retry(e, attempt)

                    time.sleep(delay)

                except KeyboardInterrupt:
                    # never retry on keyboard interrupt
                    raise

                except Exception as e:
                    # non-retryable exception, fail immediately
                    logger.warning(f"[retry] {func.__name__} non-retryable error: {e}")
                    raise

            # should never reach here, but just in case
            if last_exception:
                raise last_exception

        return wrapper
    return decorator


class ResilientAPIClient:
    """
    base class for resilient API clients.
    provides retry, circuit breaker, and rate limiting.
    """

    def __init__(
        self,
        name: str,
        retry_config: Optional[RetryConfig] = None,
        circuit_config: Optional[CircuitBreakerConfig] = None
    ):
        self.name = name
        self.retry_config = retry_config or RetryConfig()
        self.circuit = CircuitBreaker(
            name=name,
            config=circuit_config or CircuitBreakerConfig()
        )

        # stats
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.retried_calls = 0

    def execute(
        self,
        operation: Callable[[], T],
        fallback: Optional[Callable[[], T]] = None,
        operation_name: str = "operation"
    ) -> Optional[T]:
        """
        execute operation with retry and circuit breaker.

        args:
            operation: the function to execute
            fallback: optional fallback if all attempts fail
            operation_name: name for logging

        returns:
            result of operation, fallback, or None
        """
        self.total_calls += 1

        # check circuit breaker
        if not self.circuit.can_execute():
            logger.warning(f"[{self.name}] circuit open, skipping {operation_name}")
            if fallback:
                return fallback()
            return None

        last_exception = None

        for attempt in range(1, self.retry_config.max_attempts + 1):
            try:
                result = operation()
                self.circuit.record_success()
                self.successful_calls += 1
                return result

            except (ConnectionError, TimeoutError, OSError) as e:
                last_exception = e
                self.retried_calls += 1

                if attempt == self.retry_config.max_attempts:
                    self.circuit.record_failure(e)
                    self.failed_calls += 1
                    logger.warning(
                        f"[{self.name}] {operation_name} failed after "
                        f"{attempt} attempts: {e}"
                    )
                    break

                # compute delay
                delay = min(
                    self.retry_config.base_delay *
                    (self.retry_config.exponential_base ** (attempt - 1)),
                    self.retry_config.max_delay
                )

                if self.retry_config.jitter:
                    import random
                    delay = delay * (0.5 + random.random())

                logger.info(
                    f"[{self.name}] {operation_name} attempt {attempt} failed, "
                    f"retrying in {delay:.1f}s"
                )
                time.sleep(delay)

            except KeyboardInterrupt:
                raise

            except Exception as e:
                # non-retryable - might be bad request, not service error
                self.circuit.record_failure(e)
                self.failed_calls += 1
                logger.warning(f"[{self.name}] {operation_name} error: {e}")
                last_exception = e
                break

        # all attempts failed, try fallback
        if fallback:
            logger.info(f"[{self.name}] using fallback for {operation_name}")
            try:
                return fallback()
            except Exception as e:
                logger.warning(f"[{self.name}] fallback also failed: {e}")

        return None

    def stats(self) -> dict:
        """get client statistics."""
        return {
            "name": self.name,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "retried_calls": self.retried_calls,
            "success_rate": (
                self.successful_calls / self.total_calls
                if self.total_calls > 0 else 0
            ),
            "circuit_state": self.circuit.state.value,
            "circuit_failures": self.circuit.failure_count
        }


class ProviderFallback:
    """
    manages fallback between multiple providers.
    tries primary first, falls back to secondary on failure.
    """

    def __init__(self, providers: List[Any]):
        """
        args:
            providers: list of provider instances, in priority order
        """
        self.providers = providers
        self.provider_stats = {p.name: {"attempts": 0, "successes": 0} for p in providers}

    def execute(
        self,
        method_name: str,
        *args,
        **kwargs
    ) -> Optional[Any]:
        """
        execute method on providers with fallback.
        tries each provider in order until one succeeds.
        """
        last_error = None

        for provider in self.providers:
            method = getattr(provider, method_name, None)
            if not method:
                continue

            self.provider_stats[provider.name]["attempts"] += 1

            try:
                result = method(*args, **kwargs)
                if result is not None:
                    self.provider_stats[provider.name]["successes"] += 1
                    return result
                # None result might be valid (not found) or might indicate failure
                # continue to next provider only if we think it's a failure

            except Exception as e:
                logger.info(
                    f"[fallback] {provider.name}.{method_name} failed: {e}, "
                    f"trying next provider"
                )
                last_error = e
                continue

        if last_error:
            logger.warning(f"[fallback] all providers failed for {method_name}")

        return None

    def stats(self) -> dict:
        """get fallback statistics."""
        return self.provider_stats


def safe_execute(
    func: Callable[[], T],
    default: T = None,
    log_errors: bool = True,
    error_msg: str = ""
) -> T:
    """
    execute function safely, returning default on any error.

    use for non-critical operations where failure is acceptable.
    """
    try:
        return func()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        if log_errors:
            msg = error_msg or f"safe_execute failed: {e}"
            logger.warning(msg)
        return default


def validate_paper_data(data: dict) -> Tuple[bool, List[str]]:
    """
    validate paper data from API response.
    returns (is_valid, list of issues).
    """
    issues = []

    # must have at least one identifier
    has_id = any([
        data.get("doi"),
        data.get("id"),
        data.get("paperId"),
        data.get("openalex_id")
    ])
    if not has_id:
        issues.append("no identifier found")

    # should have title
    if not data.get("title"):
        issues.append("missing title")

    # year should be reasonable if present
    year = data.get("year") or data.get("publication_year")
    if year:
        try:
            year = int(year)
            if year < 1800 or year > 2030:
                issues.append(f"suspicious year: {year}")
        except (ValueError, TypeError):
            issues.append(f"invalid year format: {year}")

    return len(issues) == 0, issues


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None
):
    """
    setup refnet logging.
    call once at startup.
    """
    formatter = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    # configure refnet logger
    logger.setLevel(level)
    logger.addHandler(console_handler)

    # file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    return logger

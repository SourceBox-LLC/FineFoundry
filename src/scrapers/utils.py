"""Shared utilities for scraper reliability.

Provides retry logic, rate limiting, and error handling for all scrapers.
"""

from __future__ import annotations

import random
import time
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

import requests

# Type variable for generic return type
T = TypeVar("T")


class RateLimiter:
    """Simple rate limiter with jitter to avoid thundering herd."""

    def __init__(
        self,
        min_delay: float = 0.5,
        max_delay: float = 2.0,
        jitter: float = 0.3,
    ):
        """Initialize rate limiter.

        Args:
            min_delay: Minimum delay between requests in seconds.
            max_delay: Maximum delay between requests in seconds.
            jitter: Random jitter factor (0.3 = ±30% variation).
        """
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self._last_request: float = 0.0

    def wait(self) -> None:
        """Wait appropriate time before next request."""
        now = time.time()
        elapsed = now - self._last_request
        base_delay = random.uniform(self.min_delay, self.max_delay)
        jitter_amount = base_delay * self.jitter * random.uniform(-1, 1)
        delay = max(0, base_delay + jitter_amount - elapsed)
        if delay > 0:
            time.sleep(delay)
        self._last_request = time.time()

    def backoff(self, seconds: float) -> None:
        """Apply additional backoff (e.g., from API response)."""
        time.sleep(seconds)
        self._last_request = time.time()


def retry_request(
    func: Callable[..., T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential: bool = True,
    retryable_status_codes: tuple[int, ...] = (429, 500, 502, 503, 504),
    retryable_exceptions: tuple[type, ...] = (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.ChunkedEncodingError,
    ),
) -> Callable[..., T]:
    """Decorator to add retry logic to request functions.

    Args:
        func: Function to wrap.
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay between retries in seconds.
        max_delay: Maximum delay between retries.
        exponential: Use exponential backoff if True.
        retryable_status_codes: HTTP status codes that should trigger retry.
        retryable_exceptions: Exception types that should trigger retry.

    Returns:
        Wrapped function with retry logic.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        last_exception: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                result = func(*args, **kwargs)
                # Check for retryable HTTP status codes
                if isinstance(result, requests.Response):
                    if result.status_code in retryable_status_codes:
                        if attempt < max_retries:
                            delay = _calculate_delay(attempt, base_delay, max_delay, exponential)
                            # Check for Retry-After header
                            retry_after = result.headers.get("Retry-After")
                            if retry_after:
                                try:
                                    delay = max(delay, float(retry_after))
                                except ValueError:
                                    pass
                            time.sleep(delay)
                            continue
                        result.raise_for_status()
                return result
            except retryable_exceptions as e:
                last_exception = e
                if attempt < max_retries:
                    delay = _calculate_delay(attempt, base_delay, max_delay, exponential)
                    time.sleep(delay)
                    continue
                raise
            except requests.exceptions.HTTPError as e:
                # Check if it's a retryable status code
                if hasattr(e, "response") and e.response is not None:
                    if e.response.status_code in retryable_status_codes:
                        last_exception = e
                        if attempt < max_retries:
                            delay = _calculate_delay(attempt, base_delay, max_delay, exponential)
                            time.sleep(delay)
                            continue
                raise
        # Should not reach here, but just in case
        if last_exception:
            raise last_exception
        raise RuntimeError("Retry logic failed unexpectedly")

    return wrapper


def _calculate_delay(attempt: int, base_delay: float, max_delay: float, exponential: bool) -> float:
    """Calculate delay for retry attempt."""
    if exponential:
        delay = base_delay * (2**attempt)
    else:
        delay = base_delay
    # Add jitter (±25%)
    jitter = delay * 0.25 * random.uniform(-1, 1)
    delay = min(delay + jitter, max_delay)
    return max(0.1, delay)


def make_request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    max_retries: int = 3,
    rate_limiter: Optional[RateLimiter] = None,
    **kwargs: Any,
) -> requests.Response:
    """Make an HTTP request with retry logic and optional rate limiting.

    Args:
        session: requests.Session to use.
        method: HTTP method (GET, POST, etc.).
        url: URL to request.
        max_retries: Maximum retry attempts.
        rate_limiter: Optional RateLimiter instance.
        **kwargs: Additional arguments passed to session.request().

    Returns:
        requests.Response object.

    Raises:
        requests.exceptions.RequestException: If all retries fail.
    """
    if rate_limiter:
        rate_limiter.wait()

    last_exception: Optional[Exception] = None
    retryable_status_codes = (429, 500, 502, 503, 504)
    retryable_exceptions = (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.ChunkedEncodingError,
    )

    for attempt in range(max_retries + 1):
        try:
            response = session.request(method, url, **kwargs)

            # Handle rate limiting (429)
            if response.status_code == 429:
                if attempt < max_retries:
                    retry_after = response.headers.get("Retry-After", "5")
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        delay = 5.0
                    if rate_limiter:
                        rate_limiter.backoff(delay)
                    else:
                        time.sleep(delay)
                    continue
                response.raise_for_status()

            # Handle other retryable status codes
            if response.status_code in retryable_status_codes:
                if attempt < max_retries:
                    delay = _calculate_delay(attempt, 1.0, 30.0, exponential=True)
                    time.sleep(delay)
                    continue
                response.raise_for_status()

            return response

        except retryable_exceptions as e:
            last_exception = e
            if attempt < max_retries:
                delay = _calculate_delay(attempt, 1.0, 30.0, exponential=True)
                time.sleep(delay)
                continue
            raise

    if last_exception:
        raise last_exception
    raise RuntimeError("Request failed unexpectedly")


# Default rate limiters for each service
RATE_LIMITERS = {
    "4chan": RateLimiter(min_delay=1.0, max_delay=2.0, jitter=0.3),
    "reddit": RateLimiter(min_delay=1.0, max_delay=3.0, jitter=0.5),
    "stackexchange": RateLimiter(min_delay=0.5, max_delay=1.5, jitter=0.2),
}


def get_rate_limiter(service: str) -> RateLimiter:
    """Get or create a rate limiter for a service."""
    if service not in RATE_LIMITERS:
        RATE_LIMITERS[service] = RateLimiter()
    return RATE_LIMITERS[service]

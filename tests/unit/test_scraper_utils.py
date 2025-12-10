"""Tests for scraper utility functions."""

import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from scrapers.utils import (
    RateLimiter,
    _calculate_delay,
    get_rate_limiter,
    make_request_with_retry,
)


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_init_defaults(self):
        """Test default initialization."""
        limiter = RateLimiter()
        assert limiter.min_delay == 0.5
        assert limiter.max_delay == 2.0
        assert limiter.jitter == 0.3

    def test_init_custom(self):
        """Test custom initialization."""
        limiter = RateLimiter(min_delay=1.0, max_delay=5.0, jitter=0.5)
        assert limiter.min_delay == 1.0
        assert limiter.max_delay == 5.0
        assert limiter.jitter == 0.5

    def test_wait_updates_last_request(self):
        """Test that wait updates last request time."""
        limiter = RateLimiter(min_delay=0.01, max_delay=0.02, jitter=0)
        before = time.time()
        limiter.wait()
        after = limiter._last_request
        assert after >= before

    def test_backoff(self):
        """Test backoff applies delay."""
        limiter = RateLimiter()
        before = time.time()
        limiter.backoff(0.01)
        after = time.time()
        assert after - before >= 0.01


class TestCalculateDelay:
    """Tests for _calculate_delay function."""

    def test_linear_delay(self):
        """Test linear delay calculation."""
        delay = _calculate_delay(0, 1.0, 30.0, exponential=False)
        assert 0.75 <= delay <= 1.25  # 1.0 ± 25% jitter

    def test_exponential_delay(self):
        """Test exponential delay calculation."""
        delay0 = _calculate_delay(0, 1.0, 30.0, exponential=True)
        delay1 = _calculate_delay(1, 1.0, 30.0, exponential=True)
        delay2 = _calculate_delay(2, 1.0, 30.0, exponential=True)
        # Each should roughly double (with jitter)
        assert delay0 < delay1 < delay2

    def test_max_delay_cap(self):
        """Test that delay is capped at max_delay."""
        delay = _calculate_delay(10, 1.0, 5.0, exponential=True)
        assert delay <= 5.0


class TestGetRateLimiter:
    """Tests for get_rate_limiter function."""

    def test_returns_existing_limiter(self):
        """Test that same limiter is returned for same service."""
        limiter1 = get_rate_limiter("4chan")
        limiter2 = get_rate_limiter("4chan")
        assert limiter1 is limiter2

    def test_creates_new_limiter(self):
        """Test that new limiter is created for unknown service."""
        limiter = get_rate_limiter("unknown_service_test")
        assert isinstance(limiter, RateLimiter)


class TestMakeRequestWithRetry:
    """Tests for make_request_with_retry function."""

    def test_successful_request(self):
        """Test successful request returns response."""
        session = MagicMock(spec=requests.Session)
        response = MagicMock()
        response.status_code = 200
        session.request.return_value = response

        result = make_request_with_retry(session, "GET", "http://example.com", max_retries=1)
        assert result == response
        session.request.assert_called_once()

    def test_retry_on_429(self):
        """Test retry on rate limit (429)."""
        session = MagicMock(spec=requests.Session)
        response_429 = MagicMock()
        response_429.status_code = 429
        response_429.headers = {"Retry-After": "0.01"}
        response_200 = MagicMock()
        response_200.status_code = 200
        session.request.side_effect = [response_429, response_200]

        result = make_request_with_retry(session, "GET", "http://example.com", max_retries=2)
        assert result == response_200
        assert session.request.call_count == 2

    def test_retry_on_500(self):
        """Test retry on server error (500)."""
        session = MagicMock(spec=requests.Session)
        response_500 = MagicMock()
        response_500.status_code = 500
        response_200 = MagicMock()
        response_200.status_code = 200
        session.request.side_effect = [response_500, response_200]

        result = make_request_with_retry(session, "GET", "http://example.com", max_retries=2)
        assert result == response_200
        assert session.request.call_count == 2

    def test_retry_on_connection_error(self):
        """Test retry on connection error."""
        session = MagicMock(spec=requests.Session)
        response_200 = MagicMock()
        response_200.status_code = 200
        session.request.side_effect = [
            requests.exceptions.ConnectionError("Connection failed"),
            response_200,
        ]

        result = make_request_with_retry(session, "GET", "http://example.com", max_retries=2)
        assert result == response_200
        assert session.request.call_count == 2

    def test_raises_after_max_retries(self):
        """Test that exception is raised after max retries."""
        session = MagicMock(spec=requests.Session)
        session.request.side_effect = requests.exceptions.ConnectionError("Connection failed")

        with pytest.raises(requests.exceptions.ConnectionError):
            make_request_with_retry(session, "GET", "http://example.com", max_retries=2)
        assert session.request.call_count == 3  # Initial + 2 retries

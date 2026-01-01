"""Tests for helpers/error_messages.py.

Tests cover:
- Network error translations
- File/storage error translations
- GPU/CUDA error translations
- Training-specific errors
- Scraping-specific errors
- Inference-specific errors
"""

from helpers.error_messages import (
    friendly_error,
    friendly_scrape_error,
    friendly_training_error,
    friendly_inference_error,
)


class TestFriendlyError:
    """Tests for the general friendly_error function."""

    def test_connection_error(self):
        """Test network connection errors."""
        error = Exception("Connection refused")
        result = friendly_error(error)
        assert "connect" in result.lower() or "internet" in result.lower()

    def test_timeout_error(self):
        """Test timeout errors."""
        error = Exception("Request timeout after 30s")
        result = friendly_error(error)
        assert "connect" in result.lower() or "internet" in result.lower()

    def test_ssl_error(self):
        """Test SSL certificate errors."""
        error = Exception("SSL certificate verify failed")
        result = friendly_error(error)
        assert "secure" in result.lower() or "ssl" in result.lower()

    def test_403_forbidden(self):
        """Test 403 forbidden errors."""
        error = Exception("HTTP 403 Forbidden")
        result = friendly_error(error)
        assert "denied" in result.lower() or "permission" in result.lower()

    def test_401_unauthorized(self):
        """Test 401 unauthorized errors."""
        error = Exception("401 Unauthorized")
        result = friendly_error(error)
        assert "authentication" in result.lower() or "token" in result.lower()

    def test_404_not_found(self):
        """Test 404 not found errors."""
        error = Exception("404 Not Found")
        result = friendly_error(error)
        assert "found" in result.lower()

    def test_rate_limit(self):
        """Test rate limit errors."""
        error = Exception("429 Too Many Requests")
        result = friendly_error(error)
        assert "wait" in result.lower() or "requests" in result.lower()

    def test_permission_denied(self):
        """Test permission denied errors."""
        error = Exception("Permission denied: /some/path")
        result = friendly_error(error)
        assert "permission" in result.lower()

    def test_disk_full(self):
        """Test disk full errors."""
        error = Exception("No space left on device")
        result = friendly_error(error)
        assert "space" in result.lower() or "disk" in result.lower()

    def test_cuda_oom(self):
        """Test CUDA out of memory errors."""
        error = Exception("CUDA out of memory")
        result = friendly_error(error)
        assert "memory" in result.lower() and "gpu" in result.lower()

    def test_cuda_not_available(self):
        """Test CUDA not available errors."""
        error = Exception("CUDA is not available")
        result = friendly_error(error)
        assert "gpu" in result.lower() or "cuda" in result.lower()

    def test_database_error(self):
        """Test database errors."""
        error = Exception("SQLite database is locked")
        result = friendly_error(error)
        assert "database" in result.lower()

    def test_json_error(self):
        """Test JSON parsing errors."""
        error = Exception("JSON decode error at position 42")
        result = friendly_error(error)
        assert "format" in result.lower() or "data" in result.lower()

    def test_with_context(self):
        """Test error with context provided."""
        error = Exception("Unknown error xyz123")
        result = friendly_error(error, context="Loading config")
        assert "loading config" in result.lower()

    def test_long_error_truncated(self):
        """Test that very long errors are truncated."""
        error = Exception("a" * 500)
        result = friendly_error(error)
        assert len(result) < 300

    def test_hugging_face_error(self):
        """Test Hugging Face specific errors."""
        error = Exception("HF_TOKEN is invalid or expired")
        result = friendly_error(error)
        assert "hugging face" in result.lower() or "token" in result.lower()

    def test_runpod_error(self):
        """Test RunPod specific errors."""
        error = Exception("RunPod API key invalid")
        result = friendly_error(error)
        assert "runpod" in result.lower()


class TestFriendlyScrapeError:
    """Tests for scraping-specific error messages."""

    def test_rate_limit_with_source(self):
        """Test rate limit error mentions the source."""
        error = Exception("429 rate limit exceeded")
        result = friendly_scrape_error(error, source="Reddit")
        assert "reddit" in result.lower() or "wait" in result.lower()

    def test_forbidden_suggests_proxy(self):
        """Test 403 errors suggest using proxy."""
        error = Exception("403 Forbidden")
        result = friendly_scrape_error(error, source="4chan")
        assert "proxy" in result.lower() or "vpn" in result.lower()

    def test_timeout_mentions_delay(self):
        """Test timeout errors mention delay setting."""
        error = Exception("Request timeout")
        result = friendly_scrape_error(error)
        assert "delay" in result.lower() or "timeout" in result.lower()

    def test_empty_results(self):
        """Test empty results error."""
        error = Exception("No threads found")
        result = friendly_scrape_error(error, source="4chan")
        assert "no data" in result.lower() or "found" in result.lower()


class TestFriendlyTrainingError:
    """Tests for training-specific error messages."""

    def test_oom_suggests_batch_size(self):
        """Test OOM errors suggest reducing batch size."""
        error = Exception("CUDA out of memory")
        result = friendly_training_error(error)
        assert "batch size" in result.lower()

    def test_nan_suggests_learning_rate(self):
        """Test NaN errors suggest learning rate adjustment."""
        error = Exception("Loss is NaN, training diverged")
        result = friendly_training_error(error)
        assert "learning rate" in result.lower()

    def test_empty_dataset(self):
        """Test empty dataset errors."""
        error = Exception("Dataset is empty, no data to train on")
        result = friendly_training_error(error)
        assert "data" in result.lower()

    def test_killed_process(self):
        """Test process killed (exit 137) errors."""
        error = Exception("Process was killed with exit code 137")
        result = friendly_training_error(error)
        assert "memory" in result.lower() or "killed" in result.lower()


class TestFriendlyInferenceError:
    """Tests for inference-specific error messages."""

    def test_adapter_load_error(self):
        """Test adapter loading errors."""
        error = Exception("Failed to load LoRA adapter")
        result = friendly_inference_error(error)
        assert "adapter" in result.lower() or "training" in result.lower()

    def test_tokenizer_error(self):
        """Test tokenizer errors."""
        error = Exception("Tokenizer mismatch")
        result = friendly_inference_error(error)
        assert "tokenizer" in result.lower()

    def test_inference_oom(self):
        """Test OOM during inference."""
        error = Exception("Out of memory during generation")
        result = friendly_inference_error(error)
        assert "memory" in result.lower()

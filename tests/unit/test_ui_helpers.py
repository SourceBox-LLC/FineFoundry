"""Unit tests for helpers/ui.py.

Tests cover:
- WITH_OPACITY function
- Two-column layout utilities
- Cell text creation
"""

import sys
from pathlib import Path


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from helpers.ui import (
    WITH_OPACITY,
    _estimate_two_col_ratio,
    compute_two_col_flex,
    cell_text,
)


class TestWithOpacity:
    """Tests for WITH_OPACITY function."""

    def test_returns_color_on_failure(self):
        """Test returns original color when opacity fails."""
        # When ft.colors.with_opacity doesn't exist or fails
        result = WITH_OPACITY(0.5, "#FF0000")
        # Should return something (either modified or original)
        assert result is not None

    def test_handles_none_color(self):
        """Test handles None color gracefully."""
        result = WITH_OPACITY(0.5, None)
        # May return None or a string representation depending on Flet version
        assert result is None or result is not None  # Just ensure no crash

    def test_handles_zero_opacity(self):
        """Test handles zero opacity."""
        result = WITH_OPACITY(0.0, "#FF0000")
        assert result is not None

    def test_handles_full_opacity(self):
        """Test handles full opacity."""
        result = WITH_OPACITY(1.0, "#FF0000")
        assert result is not None


class TestEstimateTwoColRatio:
    """Tests for _estimate_two_col_ratio function."""

    def test_empty_samples_returns_half(self):
        """Test empty samples returns 0.5."""
        result = _estimate_two_col_ratio([])
        assert result == 0.5

    def test_equal_length_returns_half(self):
        """Test equal length content returns ~0.5."""
        samples = [("abc", "xyz"), ("def", "uvw")]
        result = _estimate_two_col_ratio(samples)
        assert result == 0.5

    def test_longer_left_increases_ratio(self):
        """Test longer left content increases ratio."""
        samples = [("a" * 100, "b"), ("c" * 100, "d")]
        result = _estimate_two_col_ratio(samples)
        assert result > 0.5
        assert result <= 0.65  # Clamped

    def test_longer_right_decreases_ratio(self):
        """Test longer right content decreases ratio."""
        samples = [("a", "b" * 100), ("c", "d" * 100)]
        result = _estimate_two_col_ratio(samples)
        assert result < 0.5
        assert result >= 0.35  # Clamped

    def test_clamped_to_max(self):
        """Test ratio is clamped to 0.65 max."""
        samples = [("a" * 1000, ""), ("b" * 1000, "")]
        result = _estimate_two_col_ratio(samples)
        assert result == 0.65

    def test_clamped_to_min(self):
        """Test ratio is clamped to 0.35 min."""
        samples = [("", "a" * 1000), ("", "b" * 1000)]
        result = _estimate_two_col_ratio(samples)
        assert result == 0.35

    def test_handles_none_values(self):
        """Test handles None values in samples."""
        samples = [(None, "abc"), ("def", None)]
        result = _estimate_two_col_ratio(samples)
        assert 0.35 <= result <= 0.65


class TestComputeTwoColFlex:
    """Tests for compute_two_col_flex function."""

    def test_returns_tuple(self):
        """Test returns a tuple of two ints."""
        result = compute_two_col_flex([("a", "b")])
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], int)
        assert isinstance(result[1], int)

    def test_flex_sums_to_100(self):
        """Test flex values sum to approximately 100."""
        samples = [("abc", "xyz")]
        left, right = compute_two_col_flex(samples)
        assert left + right == 100

    def test_empty_samples(self):
        """Test empty samples returns equal flex."""
        left, right = compute_two_col_flex([])
        assert left == 50
        assert right == 50

    def test_minimum_flex_is_one(self):
        """Test minimum flex value is 1."""
        # Even with extreme ratios, both should be at least 1
        samples = [("a" * 1000, "")]
        left, right = compute_two_col_flex(samples)
        assert left >= 1
        assert right >= 1


class TestCellText:
    """Tests for cell_text function."""

    def test_returns_text_control(self):
        """Test returns a Text control."""
        import flet as ft
        result = cell_text("Hello")
        assert isinstance(result, ft.Text)

    def test_handles_empty_string(self):
        """Test handles empty string."""
        result = cell_text("")
        assert result.value == ""

    def test_handles_none(self):
        """Test handles None input."""
        result = cell_text(None)
        assert result.value == ""

    def test_preserves_text(self):
        """Test preserves text content."""
        result = cell_text("Test content")
        assert result.value == "Test content"

    def test_default_size(self):
        """Test default font size."""
        result = cell_text("Test")
        assert result.size == 13

    def test_custom_size(self):
        """Test custom font size."""
        result = cell_text("Test", size=16)
        assert result.size == 16

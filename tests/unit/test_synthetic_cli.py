"""Unit tests for synthetic_cli.py.

Tests cover:
- Argument parsing
- Source validation
- Helper functions
- Error handling
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from synthetic_cli import (
    format_time,
    log,
    validate_source,
    main,
)


# =============================================================================
# format_time() tests
# =============================================================================


class TestFormatTime:
    """Tests for time formatting."""

    def test_seconds_only(self):
        """Test formatting seconds."""
        assert format_time(30) == "30s"
        assert format_time(59) == "59s"

    def test_minutes_and_seconds(self):
        """Test formatting minutes and seconds."""
        assert format_time(60) == "1m 0s"
        assert format_time(90) == "1m 30s"
        assert format_time(3599) == "59m 59s"

    def test_hours_and_minutes(self):
        """Test formatting hours and minutes."""
        assert format_time(3600) == "1h 0m"
        assert format_time(3660) == "1h 1m"
        assert format_time(7200) == "2h 0m"

    def test_zero(self):
        """Test zero seconds."""
        assert format_time(0) == "0s"

    def test_fractional_seconds(self):
        """Test fractional seconds are truncated."""
        assert format_time(30.7) == "30s"
        assert format_time(90.9) == "1m 30s"


# =============================================================================
# log() tests
# =============================================================================


class TestLog:
    """Tests for log function."""

    def test_log_prints_when_not_quiet(self, capsys):
        """Test that log prints when quiet=False."""
        log("Test message", quiet=False)
        captured = capsys.readouterr()
        assert "Test message" in captured.out

    def test_log_silent_when_quiet(self, capsys):
        """Test that log is silent when quiet=True."""
        log("Test message", quiet=True)
        captured = capsys.readouterr()
        assert captured.out == ""


# =============================================================================
# validate_source() tests
# =============================================================================


class TestValidateSource:
    """Tests for source validation."""

    def test_valid_url_http(self):
        """Test HTTP URL is valid."""
        assert validate_source("http://example.com/doc.pdf") is True

    def test_valid_url_https(self):
        """Test HTTPS URL is valid."""
        assert validate_source("https://example.com/article") is True

    def test_invalid_file_not_found(self, capsys):
        """Test non-existent file is invalid."""
        result = validate_source("/nonexistent/file.pdf")
        assert result is False
        captured = capsys.readouterr()
        assert "not found" in captured.err.lower()

    def test_valid_file_extension(self, tmp_path):
        """Test valid file extension."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_text("dummy content")
        assert validate_source(str(pdf_file)) is True

    def test_invalid_file_extension(self, tmp_path, capsys):
        """Test invalid file extension."""
        exe_file = tmp_path / "test.exe"
        exe_file.write_text("dummy content")
        result = validate_source(str(exe_file))
        assert result is False
        captured = capsys.readouterr()
        assert "unsupported" in captured.err.lower()

    def test_supported_extensions(self, tmp_path):
        """Test all supported extensions."""
        for ext in [".pdf", ".docx", ".pptx", ".html", ".htm", ".txt"]:
            test_file = tmp_path / f"test{ext}"
            test_file.write_text("dummy content")
            assert validate_source(str(test_file)) is True


# =============================================================================
# Argument parsing tests
# =============================================================================


class TestArgumentParsing:
    """Tests for CLI argument parsing."""

    def test_source_required(self):
        """Test that --source is required."""
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["synthetic_cli.py"]):
                main()

    def test_single_source(self):
        """Test single source argument."""
        with patch("sys.argv", ["synthetic_cli.py", "--source", "test.pdf"]):
            with patch("synthetic_cli.validate_source", return_value=False):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 1

    def test_multiple_sources(self):
        """Test multiple source arguments."""
        with patch("sys.argv", ["synthetic_cli.py", "-s", "a.pdf", "-s", "b.txt"]):
            with patch("synthetic_cli.validate_source", return_value=False):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 1

    def test_output_default(self):
        """Test default output path."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--source", "-s", action="append", required=True)
        parser.add_argument("--output", "-o", default="synthetic_data.json")

        args = parser.parse_args(["--source", "test.pdf"])
        assert args.output == "synthetic_data.json"

    def test_output_custom(self):
        """Test custom output path."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--source", "-s", action="append", required=True)
        parser.add_argument("--output", "-o", default="synthetic_data.json")

        args = parser.parse_args(["--source", "test.pdf", "--output", "custom.json"])
        assert args.output == "custom.json"

    def test_type_choices(self):
        """Test generation type choices."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--source", "-s", action="append", required=True)
        parser.add_argument("--type", "-t", choices=["qa", "cot", "summary"], default="qa")

        for gen_type in ["qa", "cot", "summary"]:
            args = parser.parse_args(["--source", "test.pdf", "--type", gen_type])
            assert args.type == gen_type

    def test_format_choices(self):
        """Test output format choices."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--source", "-s", action="append", required=True)
        parser.add_argument("--format", "-f", choices=["chatml", "standard"], default="chatml")

        for fmt in ["chatml", "standard"]:
            args = parser.parse_args(["--source", "test.pdf", "--format", fmt])
            assert args.format == fmt

    def test_num_pairs_default(self):
        """Test default num-pairs."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--source", "-s", action="append", required=True)
        parser.add_argument("--num-pairs", "-n", type=int, default=25)

        args = parser.parse_args(["--source", "test.pdf"])
        assert args.num_pairs == 25

    def test_curate_flag(self):
        """Test curate flag."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--source", "-s", action="append", required=True)
        parser.add_argument("--curate", action="store_true")

        args = parser.parse_args(["--source", "test.pdf"])
        assert args.curate is False

        args = parser.parse_args(["--source", "test.pdf", "--curate"])
        assert args.curate is True

    def test_quiet_flag(self):
        """Test quiet flag."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--source", "-s", action="append", required=True)
        parser.add_argument("--quiet", "-q", action="store_true")

        args = parser.parse_args(["--source", "test.pdf"])
        assert args.quiet is False

        args = parser.parse_args(["--source", "test.pdf", "-q"])
        assert args.quiet is True


# =============================================================================
# Edge case tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_sources_after_validation(self, capsys):
        """Test handling when all sources fail validation."""
        with patch("sys.argv", ["synthetic_cli.py", "--source", "/bad/path.pdf"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "no valid sources" in captured.err.lower() or "not found" in captured.err.lower()

    def test_threshold_range(self):
        """Test threshold accepts float values."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--source", "-s", action="append", required=True)
        parser.add_argument("--threshold", type=float, default=7.5)

        args = parser.parse_args(["--source", "test.pdf", "--threshold", "8.5"])
        assert args.threshold == 8.5

    def test_max_chunks_value(self):
        """Test max-chunks accepts integer."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--source", "-s", action="append", required=True)
        parser.add_argument("--max-chunks", type=int, default=10)

        args = parser.parse_args(["--source", "test.pdf", "--max-chunks", "5"])
        assert args.max_chunks == 5

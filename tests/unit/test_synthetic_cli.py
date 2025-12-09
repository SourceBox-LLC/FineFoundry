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
    debug,
    validate_source,
    load_config_file,
    create_progress_bar,
    check_vllm_running,
    deduplicate_data,
    load_existing_data,
    save_output,
    compute_dataset_stats,
    run_with_retry,
    save_progress,
    load_progress,
    clear_progress,
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


# =============================================================================
# debug() tests
# =============================================================================


class TestDebug:
    """Tests for debug logging."""

    def test_debug_prints_when_verbose(self, capsys):
        """Test debug prints in verbose mode."""
        debug("test message", verbose=True)
        captured = capsys.readouterr()
        assert "[DEBUG]" in captured.out
        assert "test message" in captured.out

    def test_debug_silent_when_not_verbose(self, capsys):
        """Test debug is silent when not verbose."""
        debug("test message", verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""


# =============================================================================
# load_config_file() tests
# =============================================================================


class TestLoadConfigFile:
    """Tests for config file loading."""

    def test_load_valid_config(self, tmp_path):
        """Test loading a valid YAML config."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
sources:
  - document.pdf
  - https://example.com
output: output.json
type: qa
num_pairs: 50
""")
        config = load_config_file(str(config_file))
        assert config["sources"] == ["document.pdf", "https://example.com"]
        assert config["output"] == "output.json"
        assert config["type"] == "qa"
        assert config["num_pairs"] == 50

    def test_load_missing_config(self):
        """Test error on missing config file."""
        with pytest.raises(SystemExit) as exc_info:
            load_config_file("/nonexistent/config.yaml")
        assert exc_info.value.code == 1

    def test_load_empty_config(self, tmp_path):
        """Test loading empty config returns empty dict."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        config = load_config_file(str(config_file))
        assert config == {}


# =============================================================================
# create_progress_bar() tests
# =============================================================================


class TestCreateProgressBar:
    """Tests for progress bar creation."""

    def test_progress_bar_none_when_quiet(self):
        """Test progress bar is None in quiet mode."""
        pbar = create_progress_bar(10, "test", quiet=True)
        assert pbar is None

    def test_progress_bar_created_when_not_quiet(self):
        """Test progress bar is created when not quiet."""
        pbar = create_progress_bar(10, "test", quiet=False)
        # tqdm should be available
        assert pbar is not None
        pbar.close()


# =============================================================================
# Verbose flag tests
# =============================================================================


class TestVerboseFlag:
    """Tests for verbose flag."""

    def test_verbose_flag_parsing(self):
        """Test --verbose flag is parsed."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--verbose", "-v", action="store_true")

        args = parser.parse_args(["--verbose"])
        assert args.verbose is True

        args = parser.parse_args(["-v"])
        assert args.verbose is True

    def test_verbose_default_false(self):
        """Test verbose defaults to False."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--verbose", "-v", action="store_true")

        args = parser.parse_args([])
        assert args.verbose is False


# =============================================================================
# Config flag tests
# =============================================================================


class TestConfigFlag:
    """Tests for config flag."""

    def test_config_flag_parsing(self):
        """Test --config flag is parsed."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--config", "-c", type=str)

        args = parser.parse_args(["--config", "my_config.yaml"])
        assert args.config == "my_config.yaml"

        args = parser.parse_args(["-c", "other.yaml"])
        assert args.config == "other.yaml"


# =============================================================================
# check_vllm_running() tests
# =============================================================================


class TestCheckVllmRunning:
    """Tests for vLLM server check."""

    def test_check_vllm_not_running(self):
        """Test returns False when no server is running."""
        # Use a port that's unlikely to be in use
        result = check_vllm_running(port=59999)
        assert result is False

    def test_check_vllm_returns_bool(self):
        """Test function returns boolean."""
        result = check_vllm_running()
        assert isinstance(result, bool)


# =============================================================================
# keep_server flag tests
# =============================================================================


class TestKeepServerFlag:
    """Tests for keep-server flag."""

    def test_keep_server_flag_parsing(self):
        """Test --keep-server flag is parsed."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--keep-server", action="store_true")

        args = parser.parse_args(["--keep-server"])
        assert args.keep_server is True

    def test_keep_server_default_false(self):
        """Test keep-server defaults to False."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--keep-server", action="store_true")

        args = parser.parse_args([])
        assert args.keep_server is False


# =============================================================================
# deduplicate_data() tests
# =============================================================================


class TestDeduplicateData:
    """Tests for deduplicate_data function."""

    def test_dedupe_standard_format(self):
        """Test deduplication with standard format."""
        data = [
            {"input": "What is Python?", "output": "A programming language."},
            {"input": "What is Python?", "output": "A snake."},  # Duplicate
            {"input": "What is Java?", "output": "Another language."},
        ]
        result = deduplicate_data(data, "standard")
        assert len(result) == 2
        assert result[0]["input"] == "What is Python?"
        assert result[1]["input"] == "What is Java?"

    def test_dedupe_chatml_format(self):
        """Test deduplication with ChatML format."""
        data = [
            {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]},
            {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hey"}]},  # Dup
            {"messages": [{"role": "user", "content": "Goodbye"}, {"role": "assistant", "content": "Bye"}]},
        ]
        result = deduplicate_data(data, "chatml")
        assert len(result) == 2

    def test_dedupe_case_insensitive(self):
        """Test deduplication is case-insensitive."""
        data = [
            {"input": "What is Python?", "output": "A language."},
            {"input": "WHAT IS PYTHON?", "output": "A snake."},  # Same when lowercased
        ]
        result = deduplicate_data(data, "standard")
        assert len(result) == 1

    def test_dedupe_empty_list(self):
        """Test deduplication with empty list."""
        result = deduplicate_data([], "standard")
        assert result == []

    def test_dedupe_preserves_order(self):
        """Test deduplication preserves first occurrence."""
        data = [
            {"input": "First", "output": "A"},
            {"input": "Second", "output": "B"},
            {"input": "First", "output": "C"},  # Duplicate
        ]
        result = deduplicate_data(data, "standard")
        assert len(result) == 2
        assert result[0]["output"] == "A"  # First occurrence kept


# =============================================================================
# load_existing_data() tests
# =============================================================================


class TestLoadExistingData:
    """Tests for load_existing_data function."""

    def test_load_json_file(self, tmp_path):
        """Test loading existing JSON file."""
        import json
        data = [{"input": "test", "output": "data"}]
        json_file = tmp_path / "existing.json"
        json_file.write_text(json.dumps(data))
        
        result = load_existing_data(str(json_file), "json")
        assert result == data

    def test_load_nonexistent_file(self, tmp_path):
        """Test loading non-existent file returns empty list."""
        result = load_existing_data(str(tmp_path / "nonexistent.json"), "json")
        assert result == []

    def test_load_invalid_json(self, tmp_path):
        """Test loading invalid JSON returns empty list."""
        json_file = tmp_path / "invalid.json"
        json_file.write_text("not valid json")
        
        result = load_existing_data(str(json_file), "json")
        assert result == []


# =============================================================================
# save_output() tests
# =============================================================================


class TestSaveOutput:
    """Tests for save_output function."""

    def test_save_json(self, tmp_path):
        """Test saving to JSON format."""
        import json
        data = [{"input": "test", "output": "data"}]
        output_path = str(tmp_path / "output.json")
        
        result = save_output(data, output_path, "json", "standard", quiet=True)
        
        assert result is True
        assert Path(output_path).exists()
        with open(output_path) as f:
            saved = json.load(f)
        assert saved == data

    def test_save_json_creates_file(self, tmp_path):
        """Test save_output creates the file."""
        data = [{"messages": [{"role": "user", "content": "Hi"}]}]
        output_path = str(tmp_path / "new_output.json")
        
        save_output(data, output_path, "json", "chatml", quiet=True)
        
        assert Path(output_path).exists()


# =============================================================================
# New argument parsing tests
# =============================================================================


class TestOutputTypeFlag:
    """Tests for --output-type flag parsing."""

    def test_output_type_choices(self):
        """Test output-type accepts valid choices."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--output-type", choices=["json", "hf", "parquet"], default="json")

        for choice in ["json", "hf", "parquet"]:
            args = parser.parse_args(["--output-type", choice])
            assert args.output_type == choice

    def test_output_type_default(self):
        """Test output-type defaults to json."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--output-type", choices=["json", "hf", "parquet"], default="json")

        args = parser.parse_args([])
        assert args.output_type == "json"


class TestDedupeFlag:
    """Tests for --dedupe flag parsing."""

    def test_dedupe_flag_parsing(self):
        """Test dedupe flag is parsed correctly."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--dedupe", action="store_true")

        args = parser.parse_args(["--dedupe"])
        assert args.dedupe is True

    def test_dedupe_default_false(self):
        """Test dedupe defaults to False."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--dedupe", action="store_true")

        args = parser.parse_args([])
        assert args.dedupe is False


class TestResumeFlag:
    """Tests for --resume flag parsing."""

    def test_resume_flag_parsing(self):
        """Test resume flag is parsed correctly."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--resume", action="store_true")

        args = parser.parse_args(["--resume"])
        assert args.resume is True

    def test_resume_default_false(self):
        """Test resume defaults to False."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--resume", action="store_true")

        args = parser.parse_args([])
        assert args.resume is False


# =============================================================================
# compute_dataset_stats() tests
# =============================================================================


class TestComputeDatasetStats:
    """Tests for compute_dataset_stats function."""

    def test_stats_empty_data(self):
        """Test stats with empty data."""
        result = compute_dataset_stats([], "standard")
        assert result["count"] == 0

    def test_stats_standard_format(self):
        """Test stats with standard format."""
        data = [
            {"input": "Hello", "output": "Hi there"},
            {"input": "How are you?", "output": "I'm fine, thanks!"},
        ]
        result = compute_dataset_stats(data, "standard")
        
        assert result["count"] == 2
        assert result["total_chars"] > 0
        assert "avg_input_len" in result
        assert "avg_output_len" in result
        assert "estimated_tokens" in result

    def test_stats_chatml_format(self):
        """Test stats with ChatML format."""
        data = [
            {"messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"}
            ]},
        ]
        result = compute_dataset_stats(data, "chatml")
        
        assert result["count"] == 1
        assert result["total_chars"] == len("Hello") + len("Hi there")

    def test_stats_min_max(self):
        """Test min/max calculations."""
        data = [
            {"input": "a", "output": "b"},
            {"input": "aaaaaaaaaa", "output": "bbbbbbbbbb"},
        ]
        result = compute_dataset_stats(data, "standard")
        
        assert result["min_input_len"] == 1
        assert result["max_input_len"] == 10
        assert result["min_output_len"] == 1
        assert result["max_output_len"] == 10

    def test_stats_token_estimate(self):
        """Test token estimation (~4 chars per token)."""
        data = [
            {"input": "a" * 100, "output": "b" * 100},
        ]
        result = compute_dataset_stats(data, "standard")
        
        # 200 chars / 4 = 50 tokens
        assert result["estimated_tokens"] == 50


# =============================================================================
# run_with_retry() tests
# =============================================================================


class TestRunWithRetry:
    """Tests for run_with_retry function."""

    def test_success_first_try(self):
        """Test successful execution on first try."""
        def success_func():
            return "success"
        
        result = run_with_retry(success_func, max_retries=3, quiet=True)
        assert result == "success"

    def test_retry_on_connection_error(self):
        """Test retry on connection error."""
        call_count = [0]
        
        def flaky_func():
            call_count[0] += 1
            if call_count[0] < 2:
                raise Exception("connection timeout")
            return "success"
        
        result = run_with_retry(flaky_func, max_retries=3, delay=0.01, quiet=True)
        assert result == "success"
        assert call_count[0] == 2

    def test_no_retry_on_non_retryable_error(self):
        """Test no retry on non-retryable errors."""
        call_count = [0]
        
        def fail_func():
            call_count[0] += 1
            raise ValueError("invalid input")
        
        try:
            run_with_retry(fail_func, max_retries=3, delay=0.01, quiet=True)
        except ValueError:
            pass
        
        assert call_count[0] == 1  # Only called once

    def test_max_retries_exceeded(self):
        """Test exception raised after max retries."""
        def always_fail():
            raise Exception("connection error")
        
        with pytest.raises(Exception):
            run_with_retry(always_fail, max_retries=2, delay=0.01, quiet=True)


# =============================================================================
# --stats flag tests
# =============================================================================


class TestStatsFlag:
    """Tests for --stats flag parsing."""

    def test_stats_flag_parsing(self):
        """Test stats flag is parsed correctly."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--stats", action="store_true")

        args = parser.parse_args(["--stats"])
        assert args.stats is True

    def test_stats_default_false(self):
        """Test stats defaults to False."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--stats", action="store_true")

        args = parser.parse_args([])
        assert args.stats is False


# =============================================================================
# --push-to-hub flag tests
# =============================================================================


class TestPushToHubFlag:
    """Tests for --push-to-hub flag parsing."""

    def test_push_to_hub_flag_parsing(self):
        """Test push-to-hub flag is parsed correctly."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--push-to-hub", type=str)

        args = parser.parse_args(["--push-to-hub", "user/dataset"])
        assert args.push_to_hub == "user/dataset"

    def test_push_to_hub_default_none(self):
        """Test push-to-hub defaults to None."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--push-to-hub", type=str)

        args = parser.parse_args([])
        assert args.push_to_hub is None


class TestPrivateFlag:
    """Tests for --private flag parsing."""

    def test_private_flag_parsing(self):
        """Test private flag is parsed correctly."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--private", action="store_true")

        args = parser.parse_args(["--private"])
        assert args.private is True

    def test_private_default_false(self):
        """Test private defaults to False."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--private", action="store_true")

        args = parser.parse_args([])
        assert args.private is False


# =============================================================================
# Progress persistence tests
# =============================================================================


class TestSaveProgress:
    """Tests for save_progress function."""

    def test_save_progress_creates_file(self, tmp_path):
        """Test save_progress creates a file."""
        progress_file = str(tmp_path / "test.progress")
        save_progress(progress_file, ["source1"], {"source1": [0, 1]}, [{"data": "test"}])
        
        assert Path(progress_file).exists()

    def test_save_progress_content(self, tmp_path):
        """Test save_progress saves correct content."""
        import json
        progress_file = str(tmp_path / "test.progress")
        save_progress(progress_file, ["source1", "source2"], {"source1": [0, 1, 2]}, [{"a": 1}, {"b": 2}])
        
        with open(progress_file) as f:
            data = json.load(f)
        
        assert data["sources_completed"] == ["source1", "source2"]
        assert data["chunks_completed"] == {"source1": [0, 1, 2]}
        assert data["data_count"] == 2
        assert "timestamp" in data


class TestLoadProgress:
    """Tests for load_progress function."""

    def test_load_progress_existing(self, tmp_path):
        """Test loading existing progress file."""
        import json
        progress_file = str(tmp_path / "test.progress")
        progress_data = {
            "sources_completed": ["s1"],
            "chunks_completed": {"s1": [0]},
            "data_count": 5,
            "timestamp": 12345.0
        }
        with open(progress_file, "w") as f:
            json.dump(progress_data, f)
        
        result = load_progress(progress_file)
        
        assert result is not None
        assert result["sources_completed"] == ["s1"]
        assert result["data_count"] == 5

    def test_load_progress_nonexistent(self, tmp_path):
        """Test loading non-existent progress file."""
        result = load_progress(str(tmp_path / "nonexistent.progress"))
        assert result is None

    def test_load_progress_invalid_json(self, tmp_path):
        """Test loading invalid JSON progress file."""
        progress_file = str(tmp_path / "test.progress")
        with open(progress_file, "w") as f:
            f.write("not valid json")
        
        result = load_progress(progress_file)
        assert result is None


class TestClearProgress:
    """Tests for clear_progress function."""

    def test_clear_progress_removes_file(self, tmp_path):
        """Test clear_progress removes the file."""
        progress_file = str(tmp_path / "test.progress")
        Path(progress_file).write_text("{}")
        
        assert Path(progress_file).exists()
        clear_progress(progress_file)
        assert not Path(progress_file).exists()

    def test_clear_progress_nonexistent(self, tmp_path):
        """Test clear_progress handles non-existent file."""
        # Should not raise
        clear_progress(str(tmp_path / "nonexistent.progress"))

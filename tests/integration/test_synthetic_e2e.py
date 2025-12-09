"""End-to-end tests for synthetic data generation pipeline.

Tests cover:
- Full generation flow with mocked model
- Output file creation
- Stats generation
- Deduplication
- Resume functionality
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from synthetic_cli import (
    deduplicate_data,
    compute_dataset_stats,
    save_output,
    load_existing_data,
    save_progress,
    load_progress,
    clear_progress,
)


class TestSyntheticE2EFlow:
    """End-to-end tests for synthetic generation flow."""

    @pytest.fixture
    def mock_generator(self):
        """Create a mock SyntheticDataKit generator."""
        mock = MagicMock()
        mock.chunk_data.return_value = ["/tmp/chunk_0.txt"]
        mock.cleanup.return_value = None
        return mock

    def test_full_dedup_stats_flow(self, tmp_path):
        """Test full flow: generate -> dedupe -> stats."""
        # Create test data with duplicates
        data = [
            {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]},
            {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hey"}]},  # Dup
            {"messages": [{"role": "user", "content": "Goodbye"}, {"role": "assistant", "content": "Bye"}]},
        ]
        
        # Deduplicate
        deduped = deduplicate_data(data, "chatml")
        assert len(deduped) == 2
        
        # Compute stats
        stats = compute_dataset_stats(deduped, "chatml")
        assert stats["count"] == 2
        assert stats["total_chars"] > 0
        assert "estimated_tokens" in stats
        
        # Save output
        output_path = str(tmp_path / "output.json")
        result = save_output(deduped, output_path, "json", "chatml", quiet=True)
        assert result is True
        assert Path(output_path).exists()
        
        # Verify saved content
        with open(output_path) as f:
            saved = json.load(f)
        assert len(saved) == 2

    def test_resume_flow(self, tmp_path):
        """Test resume flow: save progress -> load -> continue."""
        progress_file = str(tmp_path / "test.progress")
        output_file = str(tmp_path / "output.json")
        
        # Initial data
        initial_data = [
            {"messages": [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"}]},
        ]
        
        # Save initial output
        save_output(initial_data, output_file, "json", "chatml", quiet=True)
        
        # Save progress
        save_progress(progress_file, ["source1.pdf"], {"source1.pdf": [0, 1]}, initial_data)
        
        # Verify progress saved
        progress = load_progress(progress_file)
        assert progress is not None
        assert "source1.pdf" in progress["sources_completed"]
        assert progress["chunks_completed"]["source1.pdf"] == [0, 1]
        
        # Load existing data (simulating resume)
        existing = load_existing_data(output_file, "json")
        assert len(existing) == 1
        
        # Add new data
        new_data = [
            {"messages": [{"role": "user", "content": "Q2"}, {"role": "assistant", "content": "A2"}]},
        ]
        combined = existing + new_data
        
        # Save combined
        save_output(combined, output_file, "json", "chatml", quiet=True)
        
        # Verify
        final = load_existing_data(output_file, "json")
        assert len(final) == 2
        
        # Clear progress on success
        clear_progress(progress_file)
        assert not Path(progress_file).exists()

    def test_standard_format_flow(self, tmp_path):
        """Test flow with standard format."""
        data = [
            {"input": "What is Python?", "output": "A programming language."},
            {"input": "What is AI?", "output": "Artificial Intelligence."},
        ]
        
        # Compute stats
        stats = compute_dataset_stats(data, "standard")
        assert stats["count"] == 2
        assert stats["avg_input_len"] > 0
        assert stats["avg_output_len"] > 0
        
        # Save
        output_path = str(tmp_path / "standard.json")
        save_output(data, output_path, "json", "standard", quiet=True)
        
        # Load and verify
        loaded = load_existing_data(output_path, "json")
        assert len(loaded) == 2
        assert loaded[0]["input"] == "What is Python?"

    def test_empty_data_handling(self, tmp_path):
        """Test handling of empty data."""
        # Empty stats
        stats = compute_dataset_stats([], "chatml")
        assert stats["count"] == 0
        
        # Empty dedupe
        deduped = deduplicate_data([], "chatml")
        assert deduped == []
        
        # Load non-existent
        loaded = load_existing_data(str(tmp_path / "nonexistent.json"), "json")
        assert loaded == []


class TestStatsAccuracy:
    """Tests for dataset statistics accuracy."""

    def test_token_estimation(self):
        """Test token estimation is reasonable."""
        # ~4 chars per token
        data = [
            {"input": "a" * 400, "output": "b" * 400},  # 800 chars = ~200 tokens
        ]
        stats = compute_dataset_stats(data, "standard")
        assert stats["estimated_tokens"] == 200

    def test_length_ranges(self):
        """Test min/max length calculations."""
        data = [
            {"input": "short", "output": "a"},
            {"input": "this is a much longer input string", "output": "medium output"},
            {"input": "mid", "output": "very very very long output string here"},
        ]
        stats = compute_dataset_stats(data, "standard")
        
        assert stats["min_input_len"] == 3  # "mid"
        assert stats["max_input_len"] == 34  # "this is a much longer input string"
        assert stats["min_output_len"] == 1  # "a"
        assert stats["max_output_len"] == 38  # "very very very long output string here"


class TestDeduplicationEdgeCases:
    """Tests for deduplication edge cases."""

    def test_dedupe_whitespace_normalization(self):
        """Test deduplication handles whitespace."""
        data = [
            {"input": "  hello  ", "output": "world"},
            {"input": "hello", "output": "universe"},  # Same after strip
        ]
        result = deduplicate_data(data, "standard")
        assert len(result) == 1

    def test_dedupe_preserves_first(self):
        """Test deduplication keeps first occurrence."""
        data = [
            {"input": "test", "output": "first"},
            {"input": "test", "output": "second"},
            {"input": "test", "output": "third"},
        ]
        result = deduplicate_data(data, "standard")
        assert len(result) == 1
        assert result[0]["output"] == "first"

    def test_dedupe_chatml_multi_turn(self):
        """Test deduplication with multi-turn ChatML."""
        data = [
            {"messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "Fine"},
            ]},
            {"messages": [
                {"role": "user", "content": "Hello"},  # Same first user message
                {"role": "assistant", "content": "Hey there"},
            ]},
        ]
        result = deduplicate_data(data, "chatml")
        assert len(result) == 1  # Deduped on first user message

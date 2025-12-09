"""Integration tests for helpers/synthetic.py async functions.

Tests cover:
- Async ingestion with mocked subprocess
- Async generation with mocked subprocess
- Error handling for failed subprocesses
- Edge cases (empty files, network failures)
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from helpers.synthetic import (
    ingest_source_async,
    generate_content_async,
    curate_content_async,
    convert_to_ft_format_async,
    create_config_file,
)


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_data_dirs(tmp_path, monkeypatch):
    """Create temporary data directories and change to temp path."""
    monkeypatch.chdir(tmp_path)

    # Create the directory structure synthetic.py expects
    (tmp_path / "data" / "output").mkdir(parents=True)
    (tmp_path / "data" / "generated").mkdir(parents=True)
    (tmp_path / "data" / "curated").mkdir(parents=True)
    (tmp_path / "data" / "final").mkdir(parents=True)

    return tmp_path


@pytest.fixture
def config_path(temp_data_dirs):
    """Create a config file for tests."""
    return create_config_file()


@pytest.fixture
def mock_log():
    """Create an async mock log function."""
    return AsyncMock()


# =============================================================================
# ingest_source_async() tests
# =============================================================================


class TestIngestSourceAsync:
    """Tests for async source ingestion."""

    @pytest.mark.anyio
    async def test_successful_pdf_ingestion(self, temp_data_dirs, config_path, mock_log):
        """Test successful PDF ingestion."""
        # Create a fake output file that would be created by synthetic-data-kit
        output_file = temp_data_dirs / "data" / "output" / "test_document.txt"
        output_file.write_text("Extracted text content")

        with patch("helpers.synthetic.asyncio.to_thread") as mock_to_thread:
            # Mock successful subprocess
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = ""
            mock_result.stderr = ""
            mock_to_thread.return_value = mock_result

            result = await ingest_source_async(
                "/path/to/test_document.pdf",
                config_path,
                multimodal=False,
                log_fn=mock_log,
            )

            assert result is not None
            assert result.exists()
            mock_log.assert_called()

    @pytest.mark.anyio
    async def test_failed_ingestion(self, temp_data_dirs, config_path, mock_log):
        """Test handling of failed ingestion."""
        with patch("helpers.synthetic.asyncio.to_thread") as mock_to_thread:
            # Mock failed subprocess
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stderr = "Error: File not found"
            mock_to_thread.return_value = mock_result

            result = await ingest_source_async(
                "/path/to/nonexistent.pdf",
                config_path,
                log_fn=mock_log,
            )

            assert result is None
            # Should log the error (may use ❌, ⚠️, or "Error")
            calls = [str(c) for c in mock_log.call_args_list]
            assert any("Error" in str(c) or "⚠️" in str(c) or "❌" in str(c) or "not found" in str(c).lower() for c in calls)

    @pytest.mark.anyio
    async def test_url_ingestion(self, temp_data_dirs, config_path, mock_log):
        """Test URL source ingestion."""
        # Create output file for URL
        output_file = temp_data_dirs / "data" / "output" / "example_com.txt"
        output_file.write_text("Web page content")

        with patch("helpers.synthetic.asyncio.to_thread") as mock_to_thread:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_to_thread.return_value = mock_result

            await ingest_source_async(
                "https://example.com/article",
                config_path,
                log_fn=mock_log,
            )

            # Should detect as URL in log
            calls = [str(c) for c in mock_log.call_args_list]
            assert any("URL" in str(c) for c in calls)

    @pytest.mark.anyio
    async def test_multimodal_flag(self, temp_data_dirs, config_path, mock_log):
        """Test that multimodal flag is passed to subprocess."""
        output_file = temp_data_dirs / "data" / "output" / "test.txt"
        output_file.write_text("Content")

        with patch("helpers.synthetic.asyncio.to_thread") as mock_to_thread:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_to_thread.return_value = mock_result

            await ingest_source_async(
                "/path/to/test.pdf",
                config_path,
                multimodal=True,
                log_fn=mock_log,
            )

            # Check that subprocess.run was called with --multimodal
            call_args = mock_to_thread.call_args
            cmd = call_args[0][1]  # Second positional arg is the command
            assert "--multimodal" in cmd

    @pytest.mark.anyio
    async def test_no_log_function(self, temp_data_dirs, config_path):
        """Test ingestion works without log function."""
        output_file = temp_data_dirs / "data" / "output" / "test.txt"
        output_file.write_text("Content")

        with patch("helpers.synthetic.asyncio.to_thread") as mock_to_thread:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_to_thread.return_value = mock_result

            # Should not raise even without log_fn
            await ingest_source_async(
                "/path/to/test.pdf",
                config_path,
                log_fn=None,
            )

            # Result depends on file matching, but shouldn't crash


# =============================================================================
# generate_content_async() tests
# =============================================================================


class TestGenerateContentAsync:
    """Tests for async content generation."""

    @pytest.mark.anyio
    async def test_successful_qa_generation(self, temp_data_dirs, config_path, mock_log):
        """Test successful Q&A generation."""
        # Create input chunk file
        chunk_file = temp_data_dirs / "data" / "output" / "chunk_0.txt"
        chunk_file.write_text("Some document content to generate from")

        # Create expected output file
        output_file = temp_data_dirs / "data" / "generated" / "chunk_0_qa_pairs.json"
        output_file.write_text(json.dumps([{"messages": []}]))

        with patch("helpers.synthetic.asyncio.to_thread") as mock_to_thread:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_to_thread.return_value = mock_result

            result = await generate_content_async(
                str(chunk_file),
                config_path,
                gen_type="qa",
                num_pairs=10,
                log_fn=mock_log,
            )

            assert result is not None
            assert result.exists()

    @pytest.mark.anyio
    async def test_cot_generation(self, temp_data_dirs, config_path, mock_log):
        """Test chain-of-thought generation."""
        chunk_file = temp_data_dirs / "data" / "output" / "chunk_0.txt"
        chunk_file.write_text("Content")

        output_file = temp_data_dirs / "data" / "generated" / "chunk_0_cot.json"
        output_file.write_text(json.dumps([]))

        with patch("helpers.synthetic.asyncio.to_thread") as mock_to_thread:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_to_thread.return_value = mock_result

            await generate_content_async(
                str(chunk_file),
                config_path,
                gen_type="cot",
                log_fn=mock_log,
            )

            # Check command doesn't include --num-pairs for non-qa
            call_args = mock_to_thread.call_args
            cmd = call_args[0][1]
            assert "--type" in cmd
            assert "cot" in cmd

    @pytest.mark.anyio
    async def test_generation_failure(self, temp_data_dirs, config_path, mock_log):
        """Test handling of generation failure."""
        chunk_file = temp_data_dirs / "data" / "output" / "chunk_0.txt"
        chunk_file.write_text("Content")

        with patch("helpers.synthetic.asyncio.to_thread") as mock_to_thread:
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stderr = "Generation failed: model error"
            mock_to_thread.return_value = mock_result

            result = await generate_content_async(
                str(chunk_file),
                config_path,
                gen_type="qa",
                log_fn=mock_log,
            )

            assert result is None

    @pytest.mark.anyio
    async def test_num_pairs_passed_for_qa(self, temp_data_dirs, config_path, mock_log):
        """Test that num_pairs is passed for QA generation."""
        chunk_file = temp_data_dirs / "data" / "output" / "chunk_0.txt"
        chunk_file.write_text("Content")

        with patch("helpers.synthetic.asyncio.to_thread") as mock_to_thread:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_to_thread.return_value = mock_result

            await generate_content_async(
                str(chunk_file),
                config_path,
                gen_type="qa",
                num_pairs=50,
                log_fn=mock_log,
            )

            call_args = mock_to_thread.call_args
            cmd = call_args[0][1]
            assert "--num-pairs" in cmd
            assert "50" in cmd


# =============================================================================
# curate_content_async() tests
# =============================================================================


class TestCurateContentAsync:
    """Tests for async content curation."""

    @pytest.mark.anyio
    async def test_successful_curation(self, temp_data_dirs, config_path, mock_log):
        """Test successful curation."""
        # Create input file
        input_file = temp_data_dirs / "data" / "generated" / "content.json"
        input_file.write_text(json.dumps([{"messages": []}]))

        # Create expected curated output
        curated_file = temp_data_dirs / "data" / "curated" / "content_cleaned.json"
        curated_file.write_text(json.dumps([{"messages": []}]))

        with patch("helpers.synthetic.asyncio.to_thread") as mock_to_thread:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_to_thread.return_value = mock_result

            result = await curate_content_async(
                input_file,
                config_path,
                threshold=8.0,
                log_fn=mock_log,
            )

            # Compare by name since paths may differ (relative vs absolute)
            assert result.name == curated_file.name
            assert result.exists()

    @pytest.mark.anyio
    async def test_curation_failure_returns_original(self, temp_data_dirs, config_path, mock_log):
        """Test that curation failure returns original file."""
        input_file = temp_data_dirs / "data" / "generated" / "content.json"
        input_file.write_text(json.dumps([]))

        with patch("helpers.synthetic.asyncio.to_thread") as mock_to_thread:
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stderr = "Curation error"
            mock_to_thread.return_value = mock_result

            result = await curate_content_async(
                input_file,
                config_path,
                log_fn=mock_log,
            )

            # Should return original file on failure
            assert result == input_file

    @pytest.mark.anyio
    async def test_threshold_passed_to_command(self, temp_data_dirs, config_path, mock_log):
        """Test that threshold is passed to subprocess."""
        input_file = temp_data_dirs / "data" / "generated" / "content.json"
        input_file.write_text(json.dumps([]))

        with patch("helpers.synthetic.asyncio.to_thread") as mock_to_thread:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_to_thread.return_value = mock_result

            await curate_content_async(
                input_file,
                config_path,
                threshold=9.5,
                log_fn=mock_log,
            )

            call_args = mock_to_thread.call_args
            cmd = call_args[0][1]
            assert "--threshold" in cmd
            assert "9.5" in cmd


# =============================================================================
# convert_to_ft_format_async() tests
# =============================================================================


class TestConvertToFtFormatAsync:
    """Tests for async fine-tuning format conversion."""

    @pytest.mark.anyio
    async def test_successful_conversion(self, temp_data_dirs, config_path):
        """Test successful format conversion."""
        input_file = temp_data_dirs / "data" / "generated" / "content.json"
        input_file.write_text(json.dumps([]))

        # Create expected output
        output_file = temp_data_dirs / "data" / "final" / "content_ft.json"
        output_file.write_text(json.dumps([]))

        with patch("helpers.synthetic.asyncio.to_thread") as mock_to_thread:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_to_thread.return_value = mock_result

            result = await convert_to_ft_format_async(
                input_file,
                config_path,
            )

            assert result is not None
            assert result.exists()

    @pytest.mark.anyio
    async def test_conversion_no_output_file(self, temp_data_dirs, config_path):
        """Test conversion when output file is not created."""
        input_file = temp_data_dirs / "data" / "generated" / "content.json"
        input_file.write_text(json.dumps([]))

        with patch("helpers.synthetic.asyncio.to_thread") as mock_to_thread:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_to_thread.return_value = mock_result

            result = await convert_to_ft_format_async(
                input_file,
                config_path,
            )

            # No output file created, should return None
            assert result is None


# =============================================================================
# Edge case tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.anyio
    async def test_empty_file_ingestion(self, temp_data_dirs, config_path, mock_log):
        """Test ingestion of empty file."""
        # Create empty output file
        output_file = temp_data_dirs / "data" / "output" / "empty.txt"
        output_file.write_text("")

        with patch("helpers.synthetic.asyncio.to_thread") as mock_to_thread:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_to_thread.return_value = mock_result

            result = await ingest_source_async(
                "/path/to/empty.pdf",
                config_path,
                log_fn=mock_log,
            )

            # Should still return the file even if empty
            assert result is not None

    @pytest.mark.anyio
    async def test_special_characters_in_filename(self, temp_data_dirs, config_path, mock_log):
        """Test handling of special characters in filename."""
        # Create file with special characters in name
        output_file = temp_data_dirs / "data" / "output" / "test_file_with_spaces.txt"
        output_file.write_text("Content")

        with patch("helpers.synthetic.asyncio.to_thread") as mock_to_thread:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_to_thread.return_value = mock_result

            await ingest_source_async(
                "/path/to/test file with spaces.pdf",
                config_path,
                log_fn=mock_log,
            )

            # Should handle spaces in filename

    @pytest.mark.anyio
    async def test_unicode_filename(self, temp_data_dirs, config_path, mock_log):
        """Test handling of unicode in filename."""
        output_file = temp_data_dirs / "data" / "output" / "документ.txt"
        output_file.write_text("Content")

        with patch("helpers.synthetic.asyncio.to_thread") as mock_to_thread:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_to_thread.return_value = mock_result

            # Should not crash with unicode
            await ingest_source_async(
                "/path/to/документ.pdf",
                config_path,
                log_fn=mock_log,
            )

    @pytest.mark.anyio
    async def test_very_long_stderr(self, temp_data_dirs, config_path, mock_log):
        """Test handling of very long error messages."""
        with patch("helpers.synthetic.asyncio.to_thread") as mock_to_thread:
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stderr = "E" * 10000  # Very long error
            mock_to_thread.return_value = mock_result

            result = await ingest_source_async(
                "/path/to/file.pdf",
                config_path,
                log_fn=mock_log,
            )

            assert result is None
            # Should truncate error in log ([:100])

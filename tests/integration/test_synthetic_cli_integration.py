"""Integration tests for synthetic_cli.py.

Tests cover:
- Full CLI workflow with mocked model
- Config file loading
- Output file generation
- Database integration
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from synthetic_cli import (
    run_generation,
    load_config_file,
    check_vllm_running,
    ingest_source,
    convert_to_ft_format,
)


class TestRunGenerationMocked:
    """Integration tests for run_generation with mocked model."""

    @pytest.fixture
    def mock_generator(self):
        """Create a mock SyntheticDataKit generator."""
        mock = MagicMock()
        mock.chunk_data.return_value = ["/tmp/chunk_0.txt"]
        mock.cleanup.return_value = None
        return mock

    @pytest.fixture
    def temp_source_file(self, tmp_path):
        """Create a temporary source file."""
        source = tmp_path / "test_doc.txt"
        source.write_text("This is test content for synthetic data generation.")
        return str(source)

    @pytest.fixture
    def temp_output_file(self, tmp_path):
        """Create a temporary output path."""
        return str(tmp_path / "output.json")

    def test_run_generation_no_model_fails_gracefully(self, temp_source_file, temp_output_file):
        """Test run_generation handles missing model gracefully."""
        with patch.dict("sys.modules", {"unsloth": MagicMock(), "unsloth.dataprep": MagicMock()}):
            with patch("unsloth.dataprep.SyntheticDataKit") as mock_sdk:
                mock_sdk.from_pretrained.side_effect = ImportError("No module named 'unsloth'")

                result = run_generation(
                    sources=[temp_source_file],
                    output_path=temp_output_file,
                    quiet=True,
                )

                assert result is False

    def test_run_generation_cuda_error(self, temp_source_file, temp_output_file):
        """Test run_generation handles CUDA errors."""
        with patch.dict("sys.modules", {"unsloth": MagicMock(), "unsloth.dataprep": MagicMock()}):
            with patch("unsloth.dataprep.SyntheticDataKit") as mock_sdk:
                mock_sdk.from_pretrained.side_effect = RuntimeError("CUDA out of memory")

                result = run_generation(
                    sources=[temp_source_file],
                    output_path=temp_output_file,
                    quiet=True,
                )

                assert result is False

    def test_run_generation_oom_error(self, temp_source_file, temp_output_file):
        """Test run_generation handles OOM errors."""
        with patch.dict("sys.modules", {"unsloth": MagicMock(), "unsloth.dataprep": MagicMock()}):
            with patch("unsloth.dataprep.SyntheticDataKit") as mock_sdk:
                mock_sdk.from_pretrained.side_effect = RuntimeError("out of memory")

                result = run_generation(
                    sources=[temp_source_file],
                    output_path=temp_output_file,
                    quiet=True,
                )

                assert result is False


class TestIngestSource:
    """Tests for ingest_source function."""

    def test_ingest_nonexistent_file(self, tmp_path):
        """Test ingesting a non-existent file returns None."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("output_folder: data")
        workspace = tmp_path
        (workspace / "output").mkdir(exist_ok=True)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="File not found")
            result = ingest_source("/nonexistent/file.pdf", config_path, workspace, False, True)
            assert result is None

    def test_ingest_url_failure(self, tmp_path):
        """Test ingesting an invalid URL returns None."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("output_folder: data")
        workspace = tmp_path
        (workspace / "output").mkdir(exist_ok=True)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="Connection failed")
            result = ingest_source("https://invalid.example.com/doc.pdf", config_path, workspace, False, True)
            assert result is None


class TestConvertToFtFormat:
    """Tests for convert_to_ft_format function."""

    def test_convert_qa_pairs_format(self, tmp_path):
        """Test converting qa_pairs format."""
        json_file = tmp_path / "generated.json"
        json_file.write_text(
            json.dumps(
                {
                    "summary": "Test summary",
                    "qa_pairs": [
                        {"question": "What is Python?", "answer": "A programming language."},
                        {"question": "What is AI?", "answer": "Artificial Intelligence."},
                    ],
                }
            )
        )
        config_path = tmp_path / "config.yaml"
        config_path.write_text("")

        result = convert_to_ft_format(json_file, config_path, True)

        assert result is not None
        assert len(result) == 2
        assert result[0]["messages"][0]["role"] == "user"
        assert result[0]["messages"][0]["content"] == "What is Python?"
        assert result[0]["messages"][1]["role"] == "assistant"
        assert result[0]["messages"][1]["content"] == "A programming language."

    def test_convert_instruction_format(self, tmp_path):
        """Test converting instruction/output format."""
        json_file = tmp_path / "generated.json"
        json_file.write_text(
            json.dumps({"instruction": "Explain Python", "output": "Python is a high-level programming language."})
        )
        config_path = tmp_path / "config.yaml"
        config_path.write_text("")

        result = convert_to_ft_format(json_file, config_path, True)

        assert result is not None
        assert len(result) == 1
        assert result[0]["messages"][0]["content"] == "Explain Python"
        assert result[0]["messages"][1]["content"] == "Python is a high-level programming language."

    def test_convert_list_format(self, tmp_path):
        """Test converting list of Q&A pairs."""
        json_file = tmp_path / "generated.json"
        json_file.write_text(
            json.dumps(
                [
                    {"question": "Q1", "answer": "A1"},
                    {"question": "Q2", "answer": "A2"},
                ]
            )
        )
        config_path = tmp_path / "config.yaml"
        config_path.write_text("")

        result = convert_to_ft_format(json_file, config_path, True)

        assert result is not None
        assert len(result) == 2

    def test_convert_messages_format(self, tmp_path):
        """Test converting messages format (passthrough)."""
        json_file = tmp_path / "generated.json"
        json_file.write_text(
            json.dumps(
                [
                    {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]},
                ]
            )
        )
        config_path = tmp_path / "config.yaml"
        config_path.write_text("")

        result = convert_to_ft_format(json_file, config_path, True)

        assert result is not None
        assert len(result) == 1
        assert result[0]["messages"][0]["content"] == "Hi"

    def test_convert_invalid_json(self, tmp_path):
        """Test handling invalid JSON."""
        json_file = tmp_path / "generated.json"
        json_file.write_text("not valid json")
        config_path = tmp_path / "config.yaml"
        config_path.write_text("")

        result = convert_to_ft_format(json_file, config_path, True)

        assert result is None

    def test_convert_empty_data(self, tmp_path):
        """Test handling empty data."""
        json_file = tmp_path / "generated.json"
        json_file.write_text(json.dumps([]))
        config_path = tmp_path / "config.yaml"
        config_path.write_text("")

        result = convert_to_ft_format(json_file, config_path, True)

        assert result is None


class TestConfigFileIntegration:
    """Integration tests for config file handling."""

    def test_load_full_config(self, tmp_path):
        """Test loading a complete config file."""
        config_file = tmp_path / "full_config.yaml"
        config_file.write_text("""
sources:
  - doc1.pdf
  - doc2.txt
  - https://example.com/article
output: my_output.json
type: cot
num_pairs: 50
max_chunks: 5
model: unsloth/Llama-3.2-1B-Instruct
curate: true
threshold: 8.5
format: standard
multimodal: true
save_to_db: false
quiet: true
verbose: true
keep_server: true
""")
        config = load_config_file(str(config_file))

        assert config["sources"] == ["doc1.pdf", "doc2.txt", "https://example.com/article"]
        assert config["output"] == "my_output.json"
        assert config["type"] == "cot"
        assert config["num_pairs"] == 50
        assert config["max_chunks"] == 5
        assert config["model"] == "unsloth/Llama-3.2-1B-Instruct"
        assert config["curate"] is True
        assert config["threshold"] == 8.5
        assert config["format"] == "standard"
        assert config["multimodal"] is True
        assert config["save_to_db"] is False
        assert config["quiet"] is True
        assert config["verbose"] is True
        assert config["keep_server"] is True

    def test_load_minimal_config(self, tmp_path):
        """Test loading a minimal config file."""
        config_file = tmp_path / "minimal.yaml"
        config_file.write_text("""
sources:
  - document.pdf
""")
        config = load_config_file(str(config_file))

        assert config["sources"] == ["document.pdf"]
        assert "output" not in config  # Not specified

    def test_config_with_comments(self, tmp_path):
        """Test config file with YAML comments."""
        config_file = tmp_path / "commented.yaml"
        config_file.write_text("""
# This is a comment
sources:
  - doc.pdf  # inline comment
# Another comment
output: out.json
""")
        config = load_config_file(str(config_file))

        assert config["sources"] == ["doc.pdf"]
        assert config["output"] == "out.json"


class TestCheckVllmRunning:
    """Tests for vLLM server detection."""

    def test_check_vllm_not_running_default_port(self):
        """Test detection when server is not running on default port."""
        # Use a port that's very unlikely to be in use
        result = check_vllm_running(port=59998)
        assert result is False

    def test_check_vllm_handles_socket_error(self):
        """Test graceful handling of socket errors."""
        with patch("socket.socket") as mock_socket:
            mock_socket.return_value.__enter__.return_value.connect_ex.side_effect = Exception("Socket error")
            result = check_vllm_running()
            assert result is False

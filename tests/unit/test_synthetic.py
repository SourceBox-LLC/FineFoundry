"""Unit tests for helpers/synthetic.py.

Tests cover:
- URL detection
- Config file creation
- Data format conversion (ChatML <-> Standard)
- Edge cases (empty data, malformed input)
"""

from helpers.synthetic import (
    SUPPORTED_EXTENSIONS,
    TYPE_SUFFIXES,
    convert_to_chatml,
    convert_to_standard,
    create_config_file,
    is_url,
)


# =============================================================================
# is_url() tests
# =============================================================================


class TestIsUrl:
    """Tests for URL detection."""

    def test_http_url(self):
        assert is_url("http://example.com") is True

    def test_https_url(self):
        assert is_url("https://example.com/path/to/doc.pdf") is True

    def test_https_with_query(self):
        assert is_url("https://example.com/page?query=1&foo=bar") is True

    def test_local_file_path_unix(self):
        assert is_url("/home/user/document.pdf") is False

    def test_local_file_path_windows(self):
        assert is_url("C:\\Users\\doc.pdf") is False

    def test_relative_path(self):
        assert is_url("./documents/file.txt") is False

    def test_just_filename(self):
        assert is_url("document.pdf") is False

    def test_empty_string(self):
        assert is_url("") is False

    def test_ftp_url_not_supported(self):
        # Only http/https are supported
        assert is_url("ftp://files.example.com/doc.pdf") is False

    def test_file_url_not_supported(self):
        assert is_url("file:///home/user/doc.pdf") is False

    def test_malformed_url(self):
        # Should not crash on malformed input
        assert is_url("not a url at all") is False

    def test_url_with_port(self):
        assert is_url("http://localhost:8080/api") is True


# =============================================================================
# create_config_file() tests
# =============================================================================


class TestCreateConfigFile:
    """Tests for config file creation."""

    def test_creates_config_file(self, tmp_path, monkeypatch):
        # Change to temp directory so config is created there
        monkeypatch.chdir(tmp_path)

        config_path = create_config_file(output_folder="test_output")

        assert config_path.exists()
        assert config_path.name == "synthetic_data_kit_config.yaml"

    def test_config_content(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        config_path = create_config_file(output_folder="my_data")
        content = config_path.read_text()

        assert "output_folder: my_data" in content
        assert "vllm:" in content
        assert "base_url:" in content
        assert "http://localhost:8000/v1" in content

    def test_config_dir_parameter(self, tmp_path):
        # Test creating config in a specific directory
        config_path = create_config_file(output_folder="test_output", config_dir=tmp_path)
        
        assert config_path.exists()
        assert config_path.parent == tmp_path
        assert config_path.name == "synthetic_data_kit_config.yaml"
        
        content = config_path.read_text()
        assert "output_folder: test_output" in content

    def test_overwrites_existing_config(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        # Create initial config
        create_config_file(output_folder="first")

        # Create again with different folder
        config_path = create_config_file(output_folder="second")
        content = config_path.read_text()

        assert "output_folder: second" in content
        assert "output_folder: first" not in content


# =============================================================================
# convert_to_chatml() tests
# =============================================================================


class TestConvertToChatML:
    """Tests for ChatML format conversion."""

    def test_basic_conversion(self):
        data = [
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ]
            }
        ]

        result = convert_to_chatml(data)

        assert len(result) == 1
        assert "messages" in result[0]
        assert len(result[0]["messages"]) == 2

    def test_preserves_messages(self):
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]
        data = [{"messages": messages}]

        result = convert_to_chatml(data)

        assert result[0]["messages"] == messages

    def test_empty_messages_filtered(self):
        data = [
            {"messages": []},  # Empty messages - filtered out (falsy)
            {
                "messages": [
                    {"role": "user", "content": "Valid"},
                    {"role": "assistant", "content": "Response"},
                ]
            },
        ]

        result = convert_to_chatml(data)

        # Empty messages list is falsy, so it's filtered out
        assert len(result) == 1
        assert result[0]["messages"][0]["content"] == "Valid"

    def test_no_messages_key_filtered(self):
        data = [
            {"other_key": "value"},  # No messages key
            {
                "messages": [
                    {"role": "user", "content": "Valid"},
                ]
            },
        ]

        result = convert_to_chatml(data)

        assert len(result) == 1
        assert "messages" in result[0]

    def test_empty_input(self):
        result = convert_to_chatml([])
        assert result == []

    def test_multiple_conversations(self):
        data = [{"messages": [{"role": "user", "content": f"Q{i}"}]} for i in range(5)]

        result = convert_to_chatml(data)

        assert len(result) == 5


# =============================================================================
# convert_to_standard() tests
# =============================================================================


class TestConvertToStandard:
    """Tests for standard input/output format conversion."""

    def test_basic_conversion(self):
        data = [
            {
                "messages": [
                    {"role": "user", "content": "What is Python?"},
                    {"role": "assistant", "content": "A programming language."},
                ]
            }
        ]

        result = convert_to_standard(data)

        assert len(result) == 1
        assert result[0]["input"] == "What is Python?"
        assert result[0]["output"] == "A programming language."

    def test_extracts_first_user_assistant_pair(self):
        data = [
            {
                "messages": [
                    {"role": "system", "content": "System prompt"},
                    {"role": "user", "content": "First question"},
                    {"role": "assistant", "content": "First answer"},
                    {"role": "user", "content": "Second question"},
                    {"role": "assistant", "content": "Second answer"},
                ]
            }
        ]

        result = convert_to_standard(data)

        # Should only get first user/assistant pair
        assert len(result) == 1
        assert result[0]["input"] == "First question"
        assert result[0]["output"] == "First answer"

    def test_missing_user_message(self):
        data = [
            {
                "messages": [
                    {"role": "assistant", "content": "Answer without question"},
                ]
            }
        ]

        result = convert_to_standard(data)

        # No user message means no valid pair
        assert len(result) == 0

    def test_missing_assistant_message(self):
        data = [
            {
                "messages": [
                    {"role": "user", "content": "Question without answer"},
                ]
            }
        ]

        result = convert_to_standard(data)

        # No assistant message means no valid pair
        assert len(result) == 0

    def test_empty_content(self):
        data = [
            {
                "messages": [
                    {"role": "user", "content": ""},
                    {"role": "assistant", "content": "Answer"},
                ]
            }
        ]

        result = convert_to_standard(data)

        # Empty user content means no valid pair
        assert len(result) == 0

    def test_empty_input(self):
        result = convert_to_standard([])
        assert result == []

    def test_no_messages_key(self):
        data = [{"input": "already standard", "output": "format"}]

        result = convert_to_standard(data)

        # No messages key means empty result
        assert len(result) == 0

    def test_multiple_valid_conversations(self):
        data = [
            {
                "messages": [
                    {"role": "user", "content": f"Question {i}"},
                    {"role": "assistant", "content": f"Answer {i}"},
                ]
            }
            for i in range(3)
        ]

        result = convert_to_standard(data)

        assert len(result) == 3
        for i, item in enumerate(result):
            assert item["input"] == f"Question {i}"
            assert item["output"] == f"Answer {i}"


# =============================================================================
# Constants tests
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_supported_extensions(self):
        assert ".pdf" in SUPPORTED_EXTENSIONS
        assert ".docx" in SUPPORTED_EXTENSIONS
        assert ".pptx" in SUPPORTED_EXTENSIONS
        assert ".html" in SUPPORTED_EXTENSIONS
        assert ".htm" in SUPPORTED_EXTENSIONS
        assert ".txt" in SUPPORTED_EXTENSIONS

    def test_unsupported_extensions(self):
        assert ".doc" not in SUPPORTED_EXTENSIONS
        assert ".xlsx" not in SUPPORTED_EXTENSIONS
        assert ".jpg" not in SUPPORTED_EXTENSIONS

    def test_type_suffixes(self):
        assert TYPE_SUFFIXES["qa"] == "_qa_pairs"
        assert TYPE_SUFFIXES["cot"] == "_cot"
        assert TYPE_SUFFIXES["summary"] == "_summary"


# =============================================================================
# Edge case tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_convert_to_standard_with_none_content(self):
        data = [
            {
                "messages": [
                    {"role": "user", "content": None},
                    {"role": "assistant", "content": "Answer"},
                ]
            }
        ]

        # Should handle None gracefully
        result = convert_to_standard(data)
        assert len(result) == 0

    def test_convert_to_standard_with_missing_content_key(self):
        data = [
            {
                "messages": [
                    {"role": "user"},  # Missing content key
                    {"role": "assistant", "content": "Answer"},
                ]
            }
        ]

        result = convert_to_standard(data)
        assert len(result) == 0

    def test_convert_to_standard_with_missing_role_key(self):
        data = [
            {
                "messages": [
                    {"content": "No role specified"},
                    {"role": "assistant", "content": "Answer"},
                ]
            }
        ]

        result = convert_to_standard(data)
        assert len(result) == 0

    def test_convert_to_chatml_with_non_list_messages(self):
        data = [{"messages": "not a list"}]

        # Should handle gracefully - messages is truthy so included
        result = convert_to_chatml(data)
        assert len(result) == 1

    def test_is_url_with_none(self):
        # Should handle None without crashing
        try:
            result = is_url(None)
            # If it doesn't crash, it should return False
            assert result is False
        except (TypeError, AttributeError):
            # Also acceptable to raise an error for None input
            pass

    def test_is_url_with_unicode(self):
        # Unicode URLs should work
        assert is_url("https://example.com/документ.pdf") is True

    def test_is_url_with_spaces(self):
        # URL with spaces (invalid but shouldn't crash)
        result = is_url("https://example.com/my document.pdf")
        # urlparse handles this, so it's still detected as URL
        assert result is True

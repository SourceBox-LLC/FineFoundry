"""Unit tests for helpers/boards.py.

Tests cover:
- Default board list
- Board loading function
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from helpers.boards import DEFAULT_BOARDS, load_4chan_boards


class TestDefaultBoards:
    """Tests for DEFAULT_BOARDS constant."""

    def test_default_boards_is_list(self):
        """Test DEFAULT_BOARDS is a list."""
        assert isinstance(DEFAULT_BOARDS, list)

    def test_default_boards_not_empty(self):
        """Test DEFAULT_BOARDS has entries."""
        assert len(DEFAULT_BOARDS) > 0

    def test_default_boards_contains_common_boards(self):
        """Test DEFAULT_BOARDS contains well-known boards."""
        common = ["a", "b", "g", "v", "pol", "sci", "fit"]
        for board in common:
            assert board in DEFAULT_BOARDS

    def test_default_boards_all_strings(self):
        """Test all entries are strings."""
        for board in DEFAULT_BOARDS:
            assert isinstance(board, str)

    def test_default_boards_no_duplicates(self):
        """Test no duplicate boards."""
        assert len(DEFAULT_BOARDS) == len(set(DEFAULT_BOARDS))


class TestLoad4chanBoards:
    """Tests for load_4chan_boards function."""

    def test_returns_list(self):
        """Test function returns a list."""
        result = load_4chan_boards()
        assert isinstance(result, list)

    def test_returns_non_empty(self):
        """Test function returns non-empty list."""
        result = load_4chan_boards()
        assert len(result) > 0

    def test_fallback_on_network_error(self):
        """Test fallback to DEFAULT_BOARDS on network error."""
        with patch("helpers.boards.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = Exception("Network error")
            result = load_4chan_boards()
            assert result == DEFAULT_BOARDS

    def test_fallback_on_timeout(self):
        """Test fallback to DEFAULT_BOARDS on timeout."""
        with patch("helpers.boards.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = TimeoutError("Connection timed out")
            result = load_4chan_boards()
            assert result == DEFAULT_BOARDS

    def test_fallback_on_invalid_json(self):
        """Test fallback on invalid JSON response."""
        with patch("helpers.boards.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_response.read.return_value = b"not valid json"
            mock_urlopen.return_value = mock_response
            result = load_4chan_boards()
            assert result == DEFAULT_BOARDS

    def test_fallback_on_empty_boards(self):
        """Test fallback when API returns empty boards list."""
        import json

        with patch("helpers.boards.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            # Return valid JSON but empty boards
            mock_response.read.return_value = json.dumps({"boards": []}).encode()
            mock_urlopen.return_value = mock_response
            result = load_4chan_boards()
            assert result == DEFAULT_BOARDS

    def test_parses_valid_api_response(self):
        """Test parsing valid API response."""
        import json

        with patch("helpers.boards.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            # Simulate API response
            api_data = {
                "boards": [
                    {"board": "a", "title": "Anime"},
                    {"board": "b", "title": "Random"},
                    {"board": "g", "title": "Technology"},
                ]
            }
            mock_response.read.return_value = json.dumps(api_data).encode()
            mock_urlopen.return_value = mock_response
            result = load_4chan_boards()
            assert result == ["a", "b", "g"]

    def test_sorts_boards_alphabetically(self):
        """Test boards are sorted alphabetically."""
        import json

        with patch("helpers.boards.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            api_data = {
                "boards": [
                    {"board": "z"},
                    {"board": "a"},
                    {"board": "m"},
                ]
            }
            mock_response.read.return_value = json.dumps(api_data).encode()
            mock_urlopen.return_value = mock_response
            result = load_4chan_boards()
            assert result == ["a", "m", "z"]

    def test_filters_none_boards(self):
        """Test None board entries are filtered out."""
        import json

        with patch("helpers.boards.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            api_data = {
                "boards": [
                    {"board": "a"},
                    {"board": None},
                    {"board": "b"},
                    {},  # Missing board key
                ]
            }
            mock_response.read.return_value = json.dumps(api_data).encode()
            mock_urlopen.return_value = mock_response
            result = load_4chan_boards()
            assert result == ["a", "b"]

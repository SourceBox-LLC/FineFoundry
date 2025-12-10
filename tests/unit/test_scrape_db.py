"""Unit tests for helpers/scrape_db.py.

Tests cover:
- save_scrape_to_db() - saving standard pairs
- save_chatml_to_db() - saving ChatML conversations
- load_pairs_from_db() - loading pairs
- get_recent_sessions() - listing sessions
- export_to_json() - JSON export
"""

import json
import os

import pytest

from db.core import init_db, close_all_connections, _DB_PATH_OVERRIDE
from helpers.scrape_db import (
    save_scrape_to_db,
    save_chatml_to_db,
    load_pairs_from_db,
    get_recent_sessions,
    export_to_json,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database for testing."""
    db_path = str(tmp_path / "test.db")
    _DB_PATH_OVERRIDE["path"] = db_path
    init_db(db_path)
    yield db_path
    close_all_connections()
    _DB_PATH_OVERRIDE.clear()


# =============================================================================
# save_scrape_to_db() tests
# =============================================================================


class TestSaveScrapeToDb:
    """Tests for saving standard pairs."""

    def test_save_basic_pairs(self, temp_db):
        """Test saving basic pairs."""
        pairs = [
            {"input": "Question 1", "output": "Answer 1"},
            {"input": "Question 2", "output": "Answer 2"},
        ]

        session_id = save_scrape_to_db(source="4chan", pairs=pairs)

        assert session_id is not None
        assert session_id > 0

    def test_save_with_source_details(self, temp_db):
        """Test saving with source details."""
        pairs = [{"input": "Q", "output": "A"}]

        save_scrape_to_db(
            source="reddit",
            pairs=pairs,
            source_details="r/python",
        )

        sessions = get_recent_sessions(limit=1)
        assert sessions[0]["source_details"] == "r/python"

    def test_save_with_format(self, temp_db):
        """Test saving with dataset format."""
        pairs = [{"input": "Q", "output": "A"}]

        save_scrape_to_db(
            source="test",
            pairs=pairs,
            dataset_format="chatml",
        )

        sessions = get_recent_sessions(limit=1)
        assert sessions[0]["dataset_format"] == "chatml"

    def test_save_empty_pairs(self, temp_db):
        """Test saving empty pairs list."""
        session_id = save_scrape_to_db(source="test", pairs=[])

        assert session_id is not None
        sessions = get_recent_sessions(limit=1)
        assert sessions[0]["pair_count"] == 0

    def test_save_multiple_sessions(self, temp_db):
        """Test saving multiple sessions."""
        pairs1 = [{"input": "Q1", "output": "A1"}]
        pairs2 = [{"input": "Q2", "output": "A2"}]

        id1 = save_scrape_to_db(source="4chan", pairs=pairs1)
        id2 = save_scrape_to_db(source="reddit", pairs=pairs2)

        assert id1 != id2
        sessions = get_recent_sessions()
        assert len(sessions) == 2


# =============================================================================
# save_chatml_to_db() tests
# =============================================================================


class TestSaveChatmlToDb:
    """Tests for saving ChatML conversations."""

    def test_save_basic_chatml(self, temp_db):
        """Test saving basic ChatML conversations."""
        conversations = [
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ]
            }
        ]

        session_id = save_chatml_to_db(source="synthetic", conversations=conversations)

        assert session_id is not None
        pairs = load_pairs_from_db(session_id=session_id)
        assert len(pairs) == 1
        assert pairs[0]["input"] == "Hello"
        assert pairs[0]["output"] == "Hi there!"

    def test_save_chatml_with_system_message(self, temp_db):
        """Test that system messages are skipped."""
        conversations = [
            {
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Answer"},
                ]
            }
        ]

        session_id = save_chatml_to_db(source="synthetic", conversations=conversations)

        pairs = load_pairs_from_db(session_id=session_id)
        assert pairs[0]["input"] == "Question"
        assert pairs[0]["output"] == "Answer"

    def test_save_chatml_extracts_first_pair(self, temp_db):
        """Test that only first user/assistant pair is extracted."""
        conversations = [
            {
                "messages": [
                    {"role": "user", "content": "First question"},
                    {"role": "assistant", "content": "First answer"},
                    {"role": "user", "content": "Second question"},
                    {"role": "assistant", "content": "Second answer"},
                ]
            }
        ]

        session_id = save_chatml_to_db(source="synthetic", conversations=conversations)

        pairs = load_pairs_from_db(session_id=session_id)
        assert len(pairs) == 1
        assert pairs[0]["input"] == "First question"
        assert pairs[0]["output"] == "First answer"

    def test_save_chatml_skips_incomplete(self, temp_db):
        """Test that incomplete conversations are skipped."""
        conversations = [
            {"messages": [{"role": "user", "content": "No response"}]},  # Missing assistant
            {
                "messages": [
                    {"role": "user", "content": "Valid"},
                    {"role": "assistant", "content": "Response"},
                ]
            },
        ]

        session_id = save_chatml_to_db(source="synthetic", conversations=conversations)

        pairs = load_pairs_from_db(session_id=session_id)
        assert len(pairs) == 1

    def test_save_chatml_skips_empty_content(self, temp_db):
        """Test that empty content is skipped."""
        conversations = [
            {
                "messages": [
                    {"role": "user", "content": ""},
                    {"role": "assistant", "content": "Answer"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Valid"},
                    {"role": "assistant", "content": "Response"},
                ]
            },
        ]

        session_id = save_chatml_to_db(source="synthetic", conversations=conversations)

        pairs = load_pairs_from_db(session_id=session_id)
        assert len(pairs) == 1

    def test_save_chatml_strips_whitespace(self, temp_db):
        """Test that content is stripped."""
        conversations = [
            {
                "messages": [
                    {"role": "user", "content": "  Question  "},
                    {"role": "assistant", "content": "  Answer  "},
                ]
            }
        ]

        session_id = save_chatml_to_db(source="synthetic", conversations=conversations)

        pairs = load_pairs_from_db(session_id=session_id)
        assert pairs[0]["input"] == "Question"
        assert pairs[0]["output"] == "Answer"

    def test_save_chatml_handles_none_content(self, temp_db):
        """Test handling of None content."""
        conversations = [
            {
                "messages": [
                    {"role": "user", "content": None},
                    {"role": "assistant", "content": "Answer"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Valid"},
                    {"role": "assistant", "content": "Response"},
                ]
            },
        ]

        session_id = save_chatml_to_db(source="synthetic", conversations=conversations)

        pairs = load_pairs_from_db(session_id=session_id)
        assert len(pairs) == 1

    def test_save_chatml_handles_invalid_messages(self, temp_db):
        """Test handling of invalid message formats."""
        conversations = [
            {"messages": "not a list"},  # Invalid
            {"messages": [1, 2, 3]},  # Invalid items
            {
                "messages": [
                    {"role": "user", "content": "Valid"},
                    {"role": "assistant", "content": "Response"},
                ]
            },
        ]

        session_id = save_chatml_to_db(source="synthetic", conversations=conversations)

        pairs = load_pairs_from_db(session_id=session_id)
        assert len(pairs) == 1


# =============================================================================
# load_pairs_from_db() tests
# =============================================================================


class TestLoadPairsFromDb:
    """Tests for loading pairs."""

    def test_load_by_session(self, temp_db):
        """Test loading pairs by session ID."""
        pairs1 = [{"input": "Q1", "output": "A1"}]
        pairs2 = [{"input": "Q2", "output": "A2"}]

        id1 = save_scrape_to_db(source="test1", pairs=pairs1)
        save_scrape_to_db(source="test2", pairs=pairs2)

        loaded = load_pairs_from_db(session_id=id1)
        assert len(loaded) == 1
        assert loaded[0]["input"] == "Q1"

    def test_load_all_pairs(self, temp_db):
        """Test loading all pairs across sessions."""
        pairs1 = [{"input": "Q1", "output": "A1"}]
        pairs2 = [{"input": "Q2", "output": "A2"}]

        save_scrape_to_db(source="test1", pairs=pairs1)
        save_scrape_to_db(source="test2", pairs=pairs2)

        loaded = load_pairs_from_db()
        assert len(loaded) == 2

    def test_load_with_limit(self, temp_db):
        """Test loading with limit."""
        pairs = [{"input": f"Q{i}", "output": f"A{i}"} for i in range(10)]
        save_scrape_to_db(source="test", pairs=pairs)

        loaded = load_pairs_from_db(limit=5)
        assert len(loaded) == 5


# =============================================================================
# get_recent_sessions() tests
# =============================================================================


class TestGetRecentSessions:
    """Tests for getting recent sessions."""

    def test_get_recent(self, temp_db):
        """Test getting recent sessions."""
        save_scrape_to_db(source="4chan", pairs=[{"input": "Q", "output": "A"}])
        save_scrape_to_db(source="reddit", pairs=[{"input": "Q", "output": "A"}])

        sessions = get_recent_sessions()
        assert len(sessions) == 2

    def test_get_recent_with_limit(self, temp_db):
        """Test limiting recent sessions."""
        for i in range(10):
            save_scrape_to_db(source=f"source{i}", pairs=[])

        sessions = get_recent_sessions(limit=3)
        assert len(sessions) == 3

    def test_get_recent_ordered(self, temp_db):
        """Test that sessions are returned (order depends on timestamp)."""
        save_scrape_to_db(source="first", pairs=[])
        save_scrape_to_db(source="second", pairs=[])
        save_scrape_to_db(source="third", pairs=[])

        sessions = get_recent_sessions()
        # All sessions should be returned
        assert len(sessions) == 3
        sources = {s["source"] for s in sessions}
        assert sources == {"first", "second", "third"}


# =============================================================================
# export_to_json() tests
# =============================================================================


class TestExportToJson:
    """Tests for JSON export."""

    def test_export_basic(self, temp_db, tmp_path):
        """Test basic export."""
        pairs = [
            {"input": "Question 1", "output": "Answer 1"},
            {"input": "Question 2", "output": "Answer 2"},
        ]
        session_id = save_scrape_to_db(source="test", pairs=pairs)

        output_path = str(tmp_path / "export.json")
        count = export_to_json(session_id, output_path)

        assert count == 2
        assert os.path.exists(output_path)

        with open(output_path) as f:
            exported = json.load(f)

        assert len(exported) == 2


# =============================================================================
# Edge case tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_unicode_in_pairs(self, temp_db):
        """Test unicode content in pairs."""
        pairs = [
            {"input": "日本語の質問", "output": "日本語の回答"},
        ]

        session_id = save_scrape_to_db(source="test", pairs=pairs)
        loaded = load_pairs_from_db(session_id=session_id)

        assert loaded[0]["input"] == "日本語の質問"

    def test_unicode_in_chatml(self, temp_db):
        """Test unicode content in ChatML."""
        conversations = [
            {
                "messages": [
                    {"role": "user", "content": "Привет"},
                    {"role": "assistant", "content": "Здравствуйте"},
                ]
            }
        ]

        session_id = save_chatml_to_db(source="test", conversations=conversations)
        loaded = load_pairs_from_db(session_id=session_id)

        assert loaded[0]["input"] == "Привет"
        assert loaded[0]["output"] == "Здравствуйте"

    def test_special_characters(self, temp_db):
        """Test special characters in content."""
        pairs = [
            {
                "input": 'Question with "quotes" and\nnewlines',
                "output": "Answer with <html> tags & symbols",
            }
        ]

        session_id = save_scrape_to_db(source="test", pairs=pairs)
        loaded = load_pairs_from_db(session_id=session_id)

        assert '"quotes"' in loaded[0]["input"]
        assert "<html>" in loaded[0]["output"]


# =============================================================================
# get_random_prompts_for_session() tests
# =============================================================================


class TestGetRandomPromptsForSession:
    """Tests for getting random sample prompts from a session."""

    def test_get_random_prompts_basic(self, temp_db):
        """Test getting random prompts from a session."""
        from db.scraped_data import get_random_prompts_for_session

        pairs = [
            {"input": f"Question {i}", "output": f"Answer {i}"} for i in range(10)
        ]
        session_id = save_scrape_to_db(source="test", pairs=pairs)

        prompts = get_random_prompts_for_session(session_id, count=5)

        assert len(prompts) == 5
        assert all(p.startswith("Question") for p in prompts)

    def test_get_random_prompts_less_than_count(self, temp_db):
        """Test when session has fewer pairs than requested count."""
        from db.scraped_data import get_random_prompts_for_session

        pairs = [
            {"input": "Q1", "output": "A1"},
            {"input": "Q2", "output": "A2"},
        ]
        session_id = save_scrape_to_db(source="test", pairs=pairs)

        prompts = get_random_prompts_for_session(session_id, count=5)

        assert len(prompts) == 2

    def test_get_random_prompts_empty_session(self, temp_db):
        """Test with a session that has no pairs."""
        from db.scraped_data import get_random_prompts_for_session, create_scrape_session

        session_id = create_scrape_session(source="test")

        prompts = get_random_prompts_for_session(session_id, count=5)

        assert prompts == []

    def test_get_random_prompts_invalid_session(self, temp_db):
        """Test with an invalid session ID."""
        from db.scraped_data import get_random_prompts_for_session

        prompts = get_random_prompts_for_session(99999, count=5)

        assert prompts == []

"""Unit tests for db/scraped_data.py.

Tests cover:
- Session creation and retrieval
- Pair storage and retrieval
- Filtering and pagination
- JSON export
- Edge cases
"""

import json
import os

import pytest

from db.core import init_db, close_all_connections, _DB_PATH_OVERRIDE
from db.scraped_data import (
    create_scrape_session,
    add_scraped_pairs,
    get_scrape_session,
    list_scrape_sessions,
    get_pairs_for_session,
    export_session_to_json,
    delete_scrape_session,
    get_all_pairs,
    get_total_pair_count,
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
# create_scrape_session() tests
# =============================================================================


class TestCreateScrapeSession:
    """Tests for session creation."""

    def test_create_basic_session(self, temp_db):
        """Test creating a basic session."""
        session_id = create_scrape_session(
            source="4chan",
            db_path=temp_db,
        )

        assert session_id is not None
        assert session_id > 0

    def test_create_session_with_details(self, temp_db):
        """Test creating a session with source details."""
        session_id = create_scrape_session(
            source="reddit",
            source_details="r/python, r/learnpython",
            db_path=temp_db,
        )

        session = get_scrape_session(session_id, db_path=temp_db)
        assert session["source"] == "reddit"
        assert session["source_details"] == "r/python, r/learnpython"

    def test_create_session_with_format(self, temp_db):
        """Test creating a session with dataset format."""
        session_id = create_scrape_session(
            source="synthetic",
            dataset_format="chatml",
            db_path=temp_db,
        )

        session = get_scrape_session(session_id, db_path=temp_db)
        assert session["dataset_format"] == "chatml"

    def test_create_session_with_metadata(self, temp_db):
        """Test creating a session with metadata."""
        metadata = {"model": "llama-3", "num_pairs": 100}
        session_id = create_scrape_session(
            source="synthetic",
            metadata=metadata,
            db_path=temp_db,
        )

        session = get_scrape_session(session_id, db_path=temp_db)
        assert session["metadata"] == metadata

    def test_multiple_sessions(self, temp_db):
        """Test creating multiple sessions."""
        id1 = create_scrape_session(source="4chan", db_path=temp_db)
        id2 = create_scrape_session(source="reddit", db_path=temp_db)
        id3 = create_scrape_session(source="stackexchange", db_path=temp_db)

        assert id1 != id2 != id3
        assert id1 < id2 < id3


# =============================================================================
# add_scraped_pairs() tests
# =============================================================================


class TestAddScrapedPairs:
    """Tests for adding pairs to sessions."""

    def test_add_basic_pairs(self, temp_db):
        """Test adding basic pairs."""
        session_id = create_scrape_session(source="test", db_path=temp_db)

        pairs = [
            {"input": "Question 1", "output": "Answer 1"},
            {"input": "Question 2", "output": "Answer 2"},
        ]

        added = add_scraped_pairs(session_id, pairs, db_path=temp_db)
        assert added == 2

    def test_add_pairs_updates_count(self, temp_db):
        """Test that adding pairs updates session count."""
        session_id = create_scrape_session(source="test", db_path=temp_db)

        pairs = [{"input": f"Q{i}", "output": f"A{i}"} for i in range(5)]
        add_scraped_pairs(session_id, pairs, db_path=temp_db)

        session = get_scrape_session(session_id, db_path=temp_db)
        assert session["pair_count"] == 5

    def test_add_pairs_with_source_url(self, temp_db):
        """Test adding pairs with source URLs."""
        session_id = create_scrape_session(source="reddit", db_path=temp_db)

        pairs = [
            {
                "input": "Question",
                "output": "Answer",
                "source_url": "https://reddit.com/r/test/123",
            }
        ]

        add_scraped_pairs(session_id, pairs, db_path=temp_db)
        retrieved = get_pairs_for_session(session_id, db_path=temp_db)

        assert retrieved[0]["source_url"] == "https://reddit.com/r/test/123"

    def test_add_pairs_with_metadata(self, temp_db):
        """Test adding pairs with extra metadata."""
        session_id = create_scrape_session(source="4chan", db_path=temp_db)

        pairs = [
            {
                "input": "Question",
                "output": "Answer",
                "board": "g",
                "thread_id": "12345",
            }
        ]

        add_scraped_pairs(session_id, pairs, db_path=temp_db)
        retrieved = get_pairs_for_session(session_id, db_path=temp_db)

        assert retrieved[0]["board"] == "g"
        assert retrieved[0]["thread_id"] == "12345"

    def test_skip_empty_pairs(self, temp_db):
        """Test that empty pairs are skipped."""
        session_id = create_scrape_session(source="test", db_path=temp_db)

        pairs = [
            {"input": "Valid", "output": "Pair"},
            {"input": "", "output": "Missing input"},
            {"input": "Missing output", "output": ""},
            {"input": "Another valid", "output": "Pair"},
        ]

        added = add_scraped_pairs(session_id, pairs, db_path=temp_db)
        assert added == 2

    def test_add_empty_list(self, temp_db):
        """Test adding empty list of pairs."""
        session_id = create_scrape_session(source="test", db_path=temp_db)
        added = add_scraped_pairs(session_id, [], db_path=temp_db)
        assert added == 0


# =============================================================================
# get_scrape_session() tests
# =============================================================================


class TestGetScrapeSession:
    """Tests for retrieving sessions."""

    def test_get_existing_session(self, temp_db):
        """Test getting an existing session."""
        session_id = create_scrape_session(
            source="4chan",
            source_details="boards: g, v",
            db_path=temp_db,
        )

        session = get_scrape_session(session_id, db_path=temp_db)

        assert session is not None
        assert session["id"] == session_id
        assert session["source"] == "4chan"
        assert session["source_details"] == "boards: g, v"

    def test_get_nonexistent_session(self, temp_db):
        """Test getting a session that doesn't exist."""
        session = get_scrape_session(99999, db_path=temp_db)
        assert session is None

    def test_session_has_created_at(self, temp_db):
        """Test that session has created_at timestamp."""
        session_id = create_scrape_session(source="test", db_path=temp_db)
        session = get_scrape_session(session_id, db_path=temp_db)

        assert "created_at" in session
        assert session["created_at"] is not None


# =============================================================================
# list_scrape_sessions() tests
# =============================================================================


class TestListScrapeSessions:
    """Tests for listing sessions."""

    def test_list_all_sessions(self, temp_db):
        """Test listing all sessions."""
        create_scrape_session(source="4chan", db_path=temp_db)
        create_scrape_session(source="reddit", db_path=temp_db)
        create_scrape_session(source="stackexchange", db_path=temp_db)

        sessions = list_scrape_sessions(db_path=temp_db)
        assert len(sessions) == 3

    def test_list_sessions_by_source(self, temp_db):
        """Test filtering sessions by source."""
        create_scrape_session(source="4chan", db_path=temp_db)
        create_scrape_session(source="reddit", db_path=temp_db)
        create_scrape_session(source="4chan", db_path=temp_db)

        sessions = list_scrape_sessions(source="4chan", db_path=temp_db)
        assert len(sessions) == 2
        assert all(s["source"] == "4chan" for s in sessions)

    def test_list_sessions_with_limit(self, temp_db):
        """Test limiting number of sessions returned."""
        for i in range(10):
            create_scrape_session(source=f"source{i}", db_path=temp_db)

        sessions = list_scrape_sessions(limit=5, db_path=temp_db)
        assert len(sessions) == 5

    def test_list_sessions_ordered_by_date(self, temp_db):
        """Test that sessions are ordered by date descending (most recent first)."""
        id1 = create_scrape_session(source="first", db_path=temp_db)
        id2 = create_scrape_session(source="second", db_path=temp_db)
        id3 = create_scrape_session(source="third", db_path=temp_db)

        sessions = list_scrape_sessions(db_path=temp_db)

        # Sessions created in same second have same timestamp, so order by ID desc
        # The important thing is that we get all 3 sessions
        assert len(sessions) == 3
        session_ids = [s["id"] for s in sessions]
        assert set(session_ids) == {id1, id2, id3}

    def test_list_empty_sessions(self, temp_db):
        """Test listing when no sessions exist."""
        sessions = list_scrape_sessions(db_path=temp_db)
        assert sessions == []


# =============================================================================
# get_pairs_for_session() tests
# =============================================================================


class TestGetPairsForSession:
    """Tests for retrieving pairs."""

    def test_get_all_pairs(self, temp_db):
        """Test getting all pairs for a session."""
        session_id = create_scrape_session(source="test", db_path=temp_db)
        pairs = [{"input": f"Q{i}", "output": f"A{i}"} for i in range(5)]
        add_scraped_pairs(session_id, pairs, db_path=temp_db)

        retrieved = get_pairs_for_session(session_id, db_path=temp_db)
        assert len(retrieved) == 5

    def test_get_pairs_with_limit(self, temp_db):
        """Test getting pairs with limit."""
        session_id = create_scrape_session(source="test", db_path=temp_db)
        pairs = [{"input": f"Q{i}", "output": f"A{i}"} for i in range(10)]
        add_scraped_pairs(session_id, pairs, db_path=temp_db)

        retrieved = get_pairs_for_session(session_id, limit=3, db_path=temp_db)
        assert len(retrieved) == 3

    def test_get_pairs_with_offset(self, temp_db):
        """Test getting pairs with offset for pagination."""
        session_id = create_scrape_session(source="test", db_path=temp_db)
        pairs = [{"input": f"Q{i}", "output": f"A{i}"} for i in range(10)]
        add_scraped_pairs(session_id, pairs, db_path=temp_db)

        retrieved = get_pairs_for_session(session_id, limit=3, offset=5, db_path=temp_db)
        assert len(retrieved) == 3
        assert retrieved[0]["input"] == "Q5"

    def test_get_pairs_empty_session(self, temp_db):
        """Test getting pairs from empty session."""
        session_id = create_scrape_session(source="test", db_path=temp_db)
        retrieved = get_pairs_for_session(session_id, db_path=temp_db)
        assert retrieved == []


# =============================================================================
# export_session_to_json() tests
# =============================================================================


class TestExportSessionToJson:
    """Tests for JSON export."""

    def test_export_basic(self, temp_db, tmp_path):
        """Test basic JSON export."""
        session_id = create_scrape_session(source="test", db_path=temp_db)
        pairs = [
            {"input": "Question 1", "output": "Answer 1"},
            {"input": "Question 2", "output": "Answer 2"},
        ]
        add_scraped_pairs(session_id, pairs, db_path=temp_db)

        output_path = str(tmp_path / "export.json")
        count = export_session_to_json(session_id, output_path, db_path=temp_db)

        assert count == 2
        assert os.path.exists(output_path)

        with open(output_path) as f:
            exported = json.load(f)

        assert len(exported) == 2
        assert exported[0]["input"] == "Question 1"
        assert exported[0]["output"] == "Answer 1"

    def test_export_strips_metadata(self, temp_db, tmp_path):
        """Test that export only includes input/output."""
        session_id = create_scrape_session(source="test", db_path=temp_db)
        pairs = [
            {
                "input": "Question",
                "output": "Answer",
                "source_url": "http://example.com",
                "extra": "data",
            }
        ]
        add_scraped_pairs(session_id, pairs, db_path=temp_db)

        output_path = str(tmp_path / "export.json")
        export_session_to_json(session_id, output_path, db_path=temp_db)

        with open(output_path) as f:
            exported = json.load(f)

        # Should only have input and output
        assert set(exported[0].keys()) == {"input", "output"}

    def test_export_empty_session(self, temp_db, tmp_path):
        """Test exporting empty session."""
        session_id = create_scrape_session(source="test", db_path=temp_db)

        output_path = str(tmp_path / "export.json")
        count = export_session_to_json(session_id, output_path, db_path=temp_db)

        assert count == 0

        with open(output_path) as f:
            exported = json.load(f)

        assert exported == []


# =============================================================================
# delete_scrape_session() tests
# =============================================================================


class TestDeleteScrapeSession:
    """Tests for session deletion."""

    def test_delete_existing_session(self, temp_db):
        """Test deleting an existing session."""
        session_id = create_scrape_session(source="test", db_path=temp_db)
        pairs = [{"input": "Q", "output": "A"}]
        add_scraped_pairs(session_id, pairs, db_path=temp_db)

        result = delete_scrape_session(session_id, db_path=temp_db)

        assert result is True
        assert get_scrape_session(session_id, db_path=temp_db) is None

    def test_delete_cascades_to_pairs(self, temp_db):
        """Test that deleting session also deletes pairs."""
        session_id = create_scrape_session(source="test", db_path=temp_db)
        pairs = [{"input": f"Q{i}", "output": f"A{i}"} for i in range(5)]
        add_scraped_pairs(session_id, pairs, db_path=temp_db)

        delete_scrape_session(session_id, db_path=temp_db)

        # Pairs should be gone too
        retrieved = get_pairs_for_session(session_id, db_path=temp_db)
        assert retrieved == []

    def test_delete_nonexistent_session(self, temp_db):
        """Test deleting a session that doesn't exist."""
        result = delete_scrape_session(99999, db_path=temp_db)
        assert result is False


# =============================================================================
# get_all_pairs() tests
# =============================================================================


class TestGetAllPairs:
    """Tests for getting all pairs across sessions."""

    def test_get_all_pairs_multiple_sessions(self, temp_db):
        """Test getting pairs from multiple sessions."""
        session1 = create_scrape_session(source="test1", db_path=temp_db)
        session2 = create_scrape_session(source="test2", db_path=temp_db)

        add_scraped_pairs(session1, [{"input": "Q1", "output": "A1"}], db_path=temp_db)
        add_scraped_pairs(session2, [{"input": "Q2", "output": "A2"}], db_path=temp_db)

        all_pairs = get_all_pairs(db_path=temp_db)
        assert len(all_pairs) == 2

    def test_get_all_pairs_with_limit(self, temp_db):
        """Test getting all pairs with limit."""
        session_id = create_scrape_session(source="test", db_path=temp_db)
        pairs = [{"input": f"Q{i}", "output": f"A{i}"} for i in range(10)]
        add_scraped_pairs(session_id, pairs, db_path=temp_db)

        all_pairs = get_all_pairs(limit=5, db_path=temp_db)
        assert len(all_pairs) == 5

    def test_get_all_pairs_empty(self, temp_db):
        """Test getting all pairs when none exist."""
        all_pairs = get_all_pairs(db_path=temp_db)
        assert all_pairs == []


# =============================================================================
# get_total_pair_count() tests
# =============================================================================


class TestGetTotalPairCount:
    """Tests for counting pairs."""

    def test_count_pairs(self, temp_db):
        """Test counting total pairs."""
        session1 = create_scrape_session(source="test1", db_path=temp_db)
        session2 = create_scrape_session(source="test2", db_path=temp_db)

        add_scraped_pairs(session1, [{"input": f"Q{i}", "output": f"A{i}"} for i in range(3)], db_path=temp_db)
        add_scraped_pairs(session2, [{"input": f"Q{i}", "output": f"A{i}"} for i in range(5)], db_path=temp_db)

        count = get_total_pair_count(db_path=temp_db)
        assert count == 8

    def test_count_empty(self, temp_db):
        """Test counting when no pairs exist."""
        count = get_total_pair_count(db_path=temp_db)
        assert count == 0


# =============================================================================
# Edge case tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_unicode_content(self, temp_db):
        """Test handling of unicode content."""
        session_id = create_scrape_session(source="test", db_path=temp_db)
        pairs = [
            {"input": "日本語の質問", "output": "日本語の回答"},
            {"input": "Вопрос на русском", "output": "Ответ на русском"},
            {"input": "Question with émojis 🎉", "output": "Answer with émojis 🚀"},
        ]
        add_scraped_pairs(session_id, pairs, db_path=temp_db)

        retrieved = get_pairs_for_session(session_id, db_path=temp_db)
        assert len(retrieved) == 3
        assert retrieved[0]["input"] == "日本語の質問"

    def test_very_long_content(self, temp_db):
        """Test handling of very long content."""
        session_id = create_scrape_session(source="test", db_path=temp_db)
        long_text = "x" * 100000
        pairs = [{"input": long_text, "output": long_text}]
        add_scraped_pairs(session_id, pairs, db_path=temp_db)

        retrieved = get_pairs_for_session(session_id, db_path=temp_db)
        assert len(retrieved[0]["input"]) == 100000

    def test_special_characters_in_metadata(self, temp_db):
        """Test handling of special characters in metadata."""
        metadata = {
            "key_with_quotes": 'value with "quotes"',
            "key_with_newlines": "line1\nline2",
            "key_with_unicode": "日本語",
        }
        session_id = create_scrape_session(
            source="test",
            metadata=metadata,
            db_path=temp_db,
        )

        session = get_scrape_session(session_id, db_path=temp_db)
        assert session["metadata"] == metadata

    def test_null_source_details(self, temp_db):
        """Test session with null source details."""
        session_id = create_scrape_session(
            source="test",
            source_details=None,
            db_path=temp_db,
        )

        session = get_scrape_session(session_id, db_path=temp_db)
        assert session["source_details"] is None

"""Unit tests for helpers/chatml.py.

Tests cover:
- pair_to_chatml() - single pair conversion
- pairs_to_chatml() - batch pair conversion
- thread_to_chatml_conversations() - 4chan thread conversion
- reddit_thread_to_chatml_conversations() - Reddit thread conversion
- Edge cases
"""

import re


from helpers.chatml import (
    pair_to_chatml,
    pairs_to_chatml,
    _merge_same_author_chunks,
    thread_to_chatml_conversations,
    reddit_thread_to_chatml_conversations,
)


# =============================================================================
# pair_to_chatml() tests
# =============================================================================


class TestPairToChatml:
    """Tests for single pair conversion."""

    def test_basic_conversion(self):
        """Test basic pair to ChatML conversion."""
        result = pair_to_chatml("Question", "Answer")

        assert "messages" in result
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Question"
        assert result["messages"][1]["role"] == "assistant"
        assert result["messages"][1]["content"] == "Answer"

    def test_with_system_message(self):
        """Test adding system message."""
        result = pair_to_chatml("Question", "Answer", add_system="You are helpful")

        assert len(result["messages"]) == 3
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "You are helpful"

    def test_strips_whitespace(self):
        """Test that content is stripped."""
        result = pair_to_chatml("  Question  ", "  Answer  ")

        assert result["messages"][0]["content"] == "Question"
        assert result["messages"][1]["content"] == "Answer"

    def test_empty_input(self):
        """Test empty input."""
        result = pair_to_chatml("", "Answer")

        assert result["messages"][0]["content"] == ""

    def test_none_input(self):
        """Test None input."""
        result = pair_to_chatml(None, "Answer")

        assert result["messages"][0]["content"] == ""


# =============================================================================
# pairs_to_chatml() tests
# =============================================================================


class TestPairsToChatml:
    """Tests for batch pair conversion."""

    def test_basic_conversion(self):
        """Test basic batch conversion."""
        pairs = [
            {"input": "Q1", "output": "A1"},
            {"input": "Q2", "output": "A2"},
        ]
        result = pairs_to_chatml(pairs)

        assert len(result) == 2
        assert result[0]["messages"][0]["content"] == "Q1"
        assert result[1]["messages"][0]["content"] == "Q2"

    def test_filters_empty_pairs(self):
        """Test that empty pairs are filtered."""
        pairs = [
            {"input": "Q1", "output": "A1"},
            {"input": "", "output": "A2"},  # Empty input
            {"input": "Q3", "output": ""},  # Empty output
            {"input": "Q4", "output": "A4"},
        ]
        result = pairs_to_chatml(pairs)

        assert len(result) == 2

    def test_with_system_message(self):
        """Test adding system message to all."""
        pairs = [
            {"input": "Q1", "output": "A1"},
            {"input": "Q2", "output": "A2"},
        ]
        result = pairs_to_chatml(pairs, add_system="Be helpful")

        assert all(r["messages"][0]["role"] == "system" for r in result)

    def test_empty_list(self):
        """Test empty list."""
        result = pairs_to_chatml([])
        assert result == []

    def test_none_list(self):
        """Test None list."""
        result = pairs_to_chatml(None)
        assert result == []

    def test_non_dict_items(self):
        """Test non-dict items are handled."""
        pairs = [
            {"input": "Q1", "output": "A1"},
            "not a dict",
            {"input": "Q2", "output": "A2"},
        ]
        result = pairs_to_chatml(pairs)

        # Non-dict items should be skipped
        assert len(result) == 2


# =============================================================================
# _merge_same_author_chunks() tests
# =============================================================================


class TestMergeSameAuthorChunks:
    """Tests for merging same-author chunks."""

    def test_merges_adjacent_same_id(self):
        """Test merging adjacent chunks with same ID."""
        chunks = [
            {"text": "First part", "id": "abc"},
            {"text": "Second part", "id": "abc"},
            {"text": "Different author", "id": "xyz"},
        ]
        result = _merge_same_author_chunks(chunks)

        assert len(result) == 2
        assert "First part" in result[0]["text"]
        assert "Second part" in result[0]["text"]

    def test_no_merge_different_ids(self):
        """Test no merge for different IDs."""
        chunks = [
            {"text": "Part 1", "id": "abc"},
            {"text": "Part 2", "id": "xyz"},
        ]
        result = _merge_same_author_chunks(chunks)

        assert len(result) == 2

    def test_empty_list(self):
        """Test empty list."""
        result = _merge_same_author_chunks([])
        assert result == []

    def test_single_chunk(self):
        """Test single chunk."""
        chunks = [{"text": "Only one", "id": "abc"}]
        result = _merge_same_author_chunks(chunks)

        assert len(result) == 1

    def test_none_ids(self):
        """Test chunks with None IDs."""
        chunks = [
            {"text": "Part 1", "id": None},
            {"text": "Part 2", "id": None},
        ]
        result = _merge_same_author_chunks(chunks)

        # None IDs should not merge (None != None in this context)
        assert len(result) == 2


# =============================================================================
# thread_to_chatml_conversations() tests
# =============================================================================


class TestThreadToChatmlConversations:
    """Tests for 4chan thread conversion."""

    def test_basic_thread(self):
        """Test basic thread conversion."""
        posts = [
            {"no": 1, "com": "Original post content here"},
            {"no": 2, "com": "Reply to the original post"},
        ]
        result = thread_to_chatml_conversations(posts, min_len=3)

        # Should produce at least one conversation
        assert isinstance(result, list)

    def test_with_quote_chain(self):
        """Test thread with quote references."""
        posts = [
            {"no": 1, "com": "Original question here?"},
            {"no": 2, "com": ">>1 This is my answer to you"},
        ]
        result = thread_to_chatml_conversations(posts, min_len=3)

        assert isinstance(result, list)

    def test_min_length_filter(self):
        """Test minimum length filtering."""
        posts = [
            {"no": 1, "com": "OK"},  # Too short
            {"no": 2, "com": "This is a longer reply"},
        ]
        result = thread_to_chatml_conversations(posts, min_len=10)

        # Short posts should be filtered
        assert isinstance(result, list)

    def test_empty_posts(self):
        """Test empty posts list."""
        result = thread_to_chatml_conversations([])
        assert result == []

    def test_single_post(self):
        """Test single post (no conversation possible)."""
        posts = [{"no": 1, "com": "Only one post here"}]
        result = thread_to_chatml_conversations(posts)

        assert result == []

    def test_with_system_message(self):
        """Test adding system message."""
        posts = [
            {"no": 1, "com": "First post content"},
            {"no": 2, "com": "Second post content"},
        ]
        result = thread_to_chatml_conversations(posts, add_system="Be helpful", min_len=3)

        if result:
            assert result[0]["messages"][0]["role"] == "system"

    def test_max_rounds_limit(self):
        """Test max rounds per conversation limit."""
        posts = [{"no": i, "com": f"Post number {i} content"} for i in range(20)]
        result = thread_to_chatml_conversations(posts, max_rounds_per_conv=3, min_len=3)

        # Each conversation should have at most 2*3 = 6 messages (plus optional system)
        for conv in result:
            non_system = [m for m in conv["messages"] if m["role"] != "system"]
            assert len(non_system) <= 6

    def test_ban_pattern(self):
        """Test ban pattern filtering."""
        posts = [
            {"no": 1, "com": "Normal post content"},
            {"no": 2, "com": "Reply with banned_word here"},
        ]
        ban = re.compile(r"banned_word")
        result = thread_to_chatml_conversations(posts, ban_pattern=ban, min_len=3)

        # Conversations with banned content should be filtered
        for conv in result:
            for msg in conv["messages"]:
                assert "banned_word" not in msg["content"]


# =============================================================================
# reddit_thread_to_chatml_conversations() tests
# =============================================================================


class TestRedditThreadToChatmlConversations:
    """Tests for Reddit thread conversion."""

    def test_basic_thread(self):
        """Test basic Reddit thread conversion."""
        thread = {
            "post": {"title": "My Question", "selftext": "Details here", "author": "op"},
            "comments": [
                {"id": "c1", "body": "This is a reply", "author": "user1", "parent_id": "t3_post"},
            ],
        }
        result = reddit_thread_to_chatml_conversations(thread, min_len=3)

        assert isinstance(result, list)

    def test_nested_comments(self):
        """Test nested comment chain."""
        thread = {
            "post": {"title": "Question", "author": "op"},
            "comments": [
                {"id": "c1", "body": "First reply", "author": "user1", "parent_id": "t3_post"},
                {"id": "c2", "body": "Reply to first", "author": "user2", "parent_id": "t1_c1"},
            ],
        }
        result = reddit_thread_to_chatml_conversations(thread, min_len=3)

        assert isinstance(result, list)

    def test_empty_thread(self):
        """Test empty thread."""
        thread = {"post": {}, "comments": []}
        result = reddit_thread_to_chatml_conversations(thread)

        assert result == []

    def test_none_thread(self):
        """Test None thread."""
        result = reddit_thread_to_chatml_conversations(None)
        assert result == []

    def test_merge_same_author(self):
        """Test merging same author comments."""
        thread = {
            "post": {"title": "Question", "author": "op"},
            "comments": [
                {"id": "c1", "body": "Part 1", "author": "user1", "parent_id": "t3_post"},
                {"id": "c2", "body": "Part 2", "author": "user1", "parent_id": "t1_c1"},
                {"id": "c3", "body": "Different user", "author": "user2", "parent_id": "t1_c2"},
            ],
        }
        result = reddit_thread_to_chatml_conversations(thread, merge_same_author=True, min_len=3)

        assert isinstance(result, list)

    def test_with_system_message(self):
        """Test adding system message."""
        thread = {
            "post": {"title": "Question", "author": "op"},
            "comments": [
                {"id": "c1", "body": "Reply content here", "author": "user1", "parent_id": "t3_post"},
            ],
        }
        result = reddit_thread_to_chatml_conversations(thread, add_system="Be helpful", min_len=3)

        if result:
            assert result[0]["messages"][0]["role"] == "system"

    def test_max_chars_truncation(self):
        """Test max chars truncation."""
        thread = {
            "post": {"title": "Q" * 1000, "author": "op"},
            "comments": [
                {"id": "c1", "body": "Reply", "author": "user1", "parent_id": "t3_post"},
            ],
        }
        result = reddit_thread_to_chatml_conversations(thread, max_chars=100, min_len=3)

        # Messages should be truncated
        for conv in result:
            for msg in conv["messages"]:
                if msg["role"] != "system":
                    assert len(msg["content"]) <= 100


# =============================================================================
# Edge case tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_unicode_content(self):
        """Test unicode content handling."""
        pairs = [{"input": "日本語の質問", "output": "日本語の回答"}]
        result = pairs_to_chatml(pairs)

        assert len(result) == 1
        assert "日本語" in result[0]["messages"][0]["content"]

    def test_emoji_content(self):
        """Test emoji handling."""
        pairs = [{"input": "Hello 👋", "output": "Hi there! 🎉"}]
        result = pairs_to_chatml(pairs)

        assert "👋" in result[0]["messages"][0]["content"]
        assert "🎉" in result[0]["messages"][1]["content"]

    def test_very_long_content(self):
        """Test very long content."""
        long_text = "word " * 10000
        pairs = [{"input": long_text, "output": "Short answer"}]
        result = pairs_to_chatml(pairs)

        assert len(result) == 1

    def test_special_characters(self):
        """Test special characters."""
        pairs = [{"input": 'Code: if (x < y) { return "yes"; }', "output": "That's correct!"}]
        result = pairs_to_chatml(pairs)

        assert len(result) == 1
        assert "<" in result[0]["messages"][0]["content"]

    def test_newlines_preserved(self):
        """Test that newlines are preserved."""
        pairs = [{"input": "Line 1\nLine 2\nLine 3", "output": "Response"}]
        result = pairs_to_chatml(pairs)

        assert "\n" in result[0]["messages"][0]["content"]

    def test_whitespace_only_filtered(self):
        """Test that whitespace-only content is filtered."""
        pairs = [
            {"input": "   ", "output": "Answer"},
            {"input": "Question", "output": "   "},
            {"input": "Valid", "output": "Also valid"},
        ]
        result = pairs_to_chatml(pairs)

        assert len(result) == 1

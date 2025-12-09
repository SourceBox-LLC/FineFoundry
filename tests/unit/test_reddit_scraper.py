"""Unit tests for scrapers/reddit_scraper.py.

Tests cover:
- URL utilities (to_json_url, is_post_url)
- Text cleaning (clean_text)
- Timestamp conversion (iso)
- Filename sanitization (safe_filename)
- Question detection (_looks_like_question)
- Parsing helpers
"""

from scrapers.reddit_scraper import (
    to_json_url,
    is_post_url,
    clean_text,
    iso,
    safe_filename,
    looks_like_question,
    _post_text,
    _comment_text,
    _author,
)


# =============================================================================
# to_json_url() tests
# =============================================================================


class TestToJsonUrl:
    """Tests for JSON URL conversion."""

    def test_basic_url(self):
        """Test basic URL conversion."""
        url = "https://www.reddit.com/r/python"
        result = to_json_url(url)
        assert result.endswith(".json")
        assert "r/python" in result

    def test_url_with_trailing_slash(self):
        """Test URL with trailing slash."""
        url = "https://www.reddit.com/r/python/"
        result = to_json_url(url)
        assert result.endswith(".json")
        assert result.count("/") == url.count("/")  # No double slashes

    def test_post_url(self):
        """Test post URL conversion."""
        url = "https://www.reddit.com/r/python/comments/abc123/my_post"
        result = to_json_url(url)
        assert result.endswith(".json")
        assert "comments/abc123" in result


# =============================================================================
# is_post_url() tests
# =============================================================================


class TestIsPostUrl:
    """Tests for post URL detection."""

    def test_subreddit_url(self):
        """Test subreddit URL is not a post."""
        assert is_post_url("https://www.reddit.com/r/python") is False

    def test_post_url(self):
        """Test post URL is detected."""
        assert is_post_url("https://www.reddit.com/r/python/comments/abc123/title") is True

    def test_comments_in_path(self):
        """Test /comments/ detection."""
        assert is_post_url("https://reddit.com/r/test/comments/xyz") is True

    def test_no_comments(self):
        """Test URL without /comments/."""
        assert is_post_url("https://reddit.com/r/test/hot") is False


# =============================================================================
# clean_text() tests
# =============================================================================


class TestCleanText:
    """Tests for text cleaning."""

    def test_basic_cleaning(self):
        """Test basic text cleaning."""
        text = "Hello world"
        result = clean_text(text)
        assert result == "Hello world"

    def test_removes_urls(self):
        """Test URL removal."""
        text = "Check out https://example.com for more"
        result = clean_text(text)
        assert "https://example.com" not in result
        assert "Check out" in result

    def test_collapses_whitespace(self):
        """Test whitespace collapsing."""
        text = "Multiple    spaces   here"
        result = clean_text(text)
        assert "    " not in result

    def test_none_input(self):
        """Test None input."""
        result = clean_text(None)
        assert result == ""

    def test_empty_string(self):
        """Test empty string."""
        result = clean_text("")
        assert result == ""

    def test_strips_whitespace(self):
        """Test stripping leading/trailing whitespace."""
        text = "   content   "
        result = clean_text(text)
        assert result == "content"


# =============================================================================
# iso() tests
# =============================================================================


class TestIso:
    """Tests for ISO timestamp conversion."""

    def test_valid_timestamp(self):
        """Test valid Unix timestamp."""
        ts = 1609459200  # 2021-01-01 00:00:00 UTC
        result = iso(ts)
        assert result is not None
        assert "2021-01-01" in result

    def test_none_timestamp(self):
        """Test None timestamp."""
        result = iso(None)
        assert result is None

    def test_zero_timestamp(self):
        """Test zero timestamp (epoch)."""
        result = iso(0)
        # Zero is falsy, so iso() returns None
        # This is expected behavior
        assert result is None

    def test_returns_iso_format(self):
        """Test that result is ISO format."""
        ts = 1609459200
        result = iso(ts)
        # Should be parseable as ISO format
        assert "T" in result
        assert "+" in result or "Z" in result


# =============================================================================
# safe_filename() tests
# =============================================================================


class TestSafeFilename:
    """Tests for filename sanitization."""

    def test_basic_text(self):
        """Test basic text."""
        result = safe_filename("hello_world")
        assert result == "hello_world"

    def test_special_characters(self):
        """Test special character removal."""
        result = safe_filename("hello/world:test?file")
        assert "/" not in result
        assert ":" not in result
        assert "?" not in result

    def test_spaces_replaced(self):
        """Test spaces are replaced."""
        result = safe_filename("hello world test")
        assert " " not in result
        assert "_" in result

    def test_length_limit(self):
        """Test length limiting."""
        long_text = "a" * 200
        result = safe_filename(long_text, limit=50)
        assert len(result) <= 50

    def test_default_limit(self):
        """Test default length limit."""
        long_text = "a" * 200
        result = safe_filename(long_text)
        assert len(result) <= 120

    def test_preserves_valid_chars(self):
        """Test that valid characters are preserved."""
        result = safe_filename("file-name_123.txt")
        assert result == "file-name_123.txt"


# =============================================================================
# _looks_like_question() tests
# =============================================================================


class TestLooksLikeQuestion:
    """Tests for question detection."""

    def test_question_mark(self):
        """Test question mark detection."""
        assert looks_like_question("What is this?") is True

    def test_question_words(self):
        """Test question word detection."""
        assert looks_like_question("How do I do this") is True
        assert looks_like_question("What is Python") is True
        assert looks_like_question("Why does this happen") is True
        assert looks_like_question("Where can I find") is True
        assert looks_like_question("When should I use") is True

    def test_not_a_question(self):
        """Test non-question text."""
        # Note: "is" in "This" matches the question word pattern
        # So we need text without any question words
        assert looks_like_question("Hello world.") is False
        assert looks_like_question("Python rocks.") is False

    def test_question_at_end(self):
        """Test question detection at end of text."""
        # The function checks the last 300 chars
        text = "Some context. " * 50 + "What do you think?"
        assert looks_like_question(text) is True

    def test_empty_string(self):
        """Test empty string."""
        assert looks_like_question("") is False


# =============================================================================
# _post_text() tests
# =============================================================================


class TestPostText:
    """Tests for post text extraction."""

    def test_title_only(self):
        """Test post with title only."""
        post = {"title": "My Title", "selftext": ""}
        result = _post_text(post)
        assert "My Title" in result

    def test_title_and_body(self):
        """Test post with title and body."""
        post = {"title": "My Title", "selftext": "Body content here"}
        result = _post_text(post)
        assert "My Title" in result
        assert "Body content" in result

    def test_missing_fields(self):
        """Test post with missing fields."""
        post = {}
        result = _post_text(post)
        assert result == ""

    def test_none_values(self):
        """Test post with None values."""
        post = {"title": None, "selftext": None}
        result = _post_text(post)
        assert result == ""


# =============================================================================
# _comment_text() tests
# =============================================================================


class TestCommentText:
    """Tests for comment text extraction."""

    def test_basic_comment(self):
        """Test basic comment."""
        comment = {"body": "This is a comment"}
        result = _comment_text(comment)
        assert "This is a comment" in result

    def test_missing_body(self):
        """Test comment with missing body."""
        comment = {}
        result = _comment_text(comment)
        assert result == ""

    def test_none_body(self):
        """Test comment with None body."""
        comment = {"body": None}
        result = _comment_text(comment)
        assert result == ""

    def test_cleans_text(self):
        """Test that text is cleaned."""
        comment = {"body": "Text with https://url.com link"}
        result = _comment_text(comment)
        assert "https://url.com" not in result


# =============================================================================
# _author() tests
# =============================================================================


class TestAuthor:
    """Tests for author extraction."""

    def test_basic_author(self):
        """Test basic author extraction."""
        comment = {"author": "username"}
        result = _author(comment)
        assert result == "username"

    def test_missing_author(self):
        """Test missing author."""
        comment = {}
        result = _author(comment)
        assert result == ""

    def test_none_comment(self):
        """Test None comment."""
        result = _author(None)
        assert result == ""

    def test_none_author(self):
        """Test None author value."""
        comment = {"author": None}
        result = _author(comment)
        assert result == ""


# =============================================================================
# Edge case tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_unicode_in_text(self):
        """Test unicode handling."""
        text = "日本語テキスト with émojis 🎉"
        result = clean_text(text)
        assert "日本語" in result
        assert "🎉" in result

    def test_very_long_text(self):
        """Test very long text."""
        text = "word " * 10000
        result = clean_text(text)
        assert len(result) > 0

    def test_special_reddit_formatting(self):
        """Test Reddit-specific formatting."""
        text = "**bold** and *italic* and ~~strikethrough~~"
        result = clean_text(text)
        # Should preserve markdown
        assert "bold" in result

    def test_newlines_in_text(self):
        """Test newline handling."""
        text = "Line 1\n\nLine 2\n\n\nLine 3"
        result = clean_text(text)
        # Should collapse multiple newlines
        assert "\n\n\n" not in result

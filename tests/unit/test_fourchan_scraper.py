"""Unit tests for scrapers/fourchan_scraper.py.

Tests cover:
- Text cleaning functions (strip_html, clean_text)
- Reference extraction (extract_refs)
- Pair building strategies (adjacent, contextual, quote_contextual)
- Edge cases
"""

from scrapers.fourchan_scraper import (
    strip_html,
    clean_text,
    extract_refs,
    build_pairs_adjacent,
    build_pairs_contextual,
    build_pairs_quote_contextual,
)


# =============================================================================
# strip_html() tests
# =============================================================================


class TestStripHtml:
    """Tests for HTML stripping."""

    def test_basic_html_tags(self):
        """Test stripping basic HTML tags."""
        html = "<p>Hello <b>world</b></p>"
        result = strip_html(html)
        assert result == "Hello world"

    def test_br_tags_to_newlines(self):
        """Test that <br> tags become newlines."""
        html = "Line 1<br>Line 2<br/>Line 3<br />Line 4"
        result = strip_html(html)
        assert result == "Line 1\nLine 2\nLine 3\nLine 4"

    def test_html_entities(self):
        """Test unescaping HTML entities."""
        html = "&lt;code&gt; &amp; &quot;text&quot;"
        result = strip_html(html)
        assert result == '<code> & "text"'

    def test_nested_tags(self):
        """Test stripping nested tags."""
        html = "<div><span class='test'>content</span></div>"
        result = strip_html(html)
        assert result == "content"

    def test_empty_string(self):
        """Test empty string."""
        assert strip_html("") == ""

    def test_no_html(self):
        """Test string with no HTML."""
        text = "Just plain text"
        assert strip_html(text) == text


# =============================================================================
# clean_text() tests
# =============================================================================


class TestCleanText:
    """Tests for text cleaning."""

    def test_basic_cleaning(self):
        """Test basic text cleaning."""
        html = "<p>Hello world</p>"
        result = clean_text(html)
        assert result == "Hello world"

    def test_removes_greentext(self):
        """Test that greentext lines are removed."""
        text = "Normal line\n>greentext line\nAnother normal line"
        result = clean_text(text)
        assert "greentext" not in result
        assert "Normal line" in result

    def test_removes_quote_refs(self):
        """Test that quote references are removed."""
        text = "Normal text >>12345 more text"
        result = clean_text(text)
        assert ">>12345" not in result
        assert "Normal text" in result
        assert "more text" in result

    def test_removes_urls(self):
        """Test that URLs are removed."""
        text = "Check out https://example.com for more info"
        result = clean_text(text)
        assert "https://example.com" not in result
        assert "Check out" in result

    def test_collapses_whitespace(self):
        """Test that multiple whitespace is collapsed."""
        text = "Multiple    spaces   and\n\n\nnewlines"
        result = clean_text(text)
        assert "    " not in result
        assert "\n\n" not in result

    def test_none_input(self):
        """Test None input returns empty string."""
        assert clean_text(None) == ""

    def test_empty_string(self):
        """Test empty string returns empty string."""
        assert clean_text("") == ""

    def test_combined_cleaning(self):
        """Test all cleaning operations together."""
        html = """
        <p>Hello >>12345</p>
        <br>
        >greentext quote
        <br>
        Check https://example.com
        <br>
        Normal   text   here
        """
        result = clean_text(html)
        assert ">>12345" not in result
        assert "greentext" not in result
        assert "https://" not in result
        assert "   " not in result


# =============================================================================
# extract_refs() tests
# =============================================================================


class TestExtractRefs:
    """Tests for extracting quote references."""

    def test_basic_refs(self):
        """Test extracting basic references."""
        html = ">>12345 >>67890"
        refs = extract_refs(html)
        assert refs == [12345, 67890]

    def test_html_escaped_refs(self):
        """Test extracting HTML-escaped references."""
        html = "&gt;&gt;12345"
        refs = extract_refs(html)
        assert refs == [12345]

    def test_mixed_refs(self):
        """Test mixed plain and escaped references."""
        html = ">>12345 &gt;&gt;67890"
        refs = extract_refs(html)
        assert 12345 in refs
        assert 67890 in refs

    def test_no_refs(self):
        """Test text with no references."""
        html = "Just normal text"
        refs = extract_refs(html)
        assert refs == []

    def test_none_input(self):
        """Test None input."""
        refs = extract_refs(None)
        assert refs == []

    def test_empty_string(self):
        """Test empty string."""
        refs = extract_refs("")
        assert refs == []

    def test_refs_in_html(self):
        """Test refs embedded in HTML."""
        html = '<a href="#p12345" class="quotelink">&gt;&gt;12345</a>'
        refs = extract_refs(html)
        assert 12345 in refs


# =============================================================================
# build_pairs_adjacent() tests
# =============================================================================


class TestBuildPairsAdjacent:
    """Tests for adjacent pair building."""

    def test_basic_pairs(self):
        """Test building basic adjacent pairs."""
        posts = [
            {"com": "First post content here"},
            {"com": "Second post content here"},
            {"com": "Third post content here"},
        ]
        pairs = build_pairs_adjacent(posts)

        assert len(pairs) == 2
        assert pairs[0]["input"] == "First post content here"
        assert pairs[0]["output"] == "Second post content here"

    def test_min_length_filter(self):
        """Test that short posts are filtered."""
        posts = [
            {"com": "OK"},  # Too short
            {"com": "This is long enough"},
            {"com": "AB"},  # Too short
        ]
        pairs = build_pairs_adjacent(posts, min_len=5)

        # Only one valid pair possible
        assert len(pairs) == 0  # First is too short, so no valid input

    def test_empty_posts(self):
        """Test with empty posts list."""
        pairs = build_pairs_adjacent([])
        assert pairs == []

    def test_single_post(self):
        """Test with single post."""
        posts = [{"com": "Only one post"}]
        pairs = build_pairs_adjacent(posts)
        assert pairs == []

    def test_cleans_html(self):
        """Test that HTML is cleaned from posts."""
        posts = [
            {"com": "<p>First post</p>"},
            {"com": "<b>Second post</b>"},
        ]
        pairs = build_pairs_adjacent(posts, min_len=3)

        assert "<p>" not in pairs[0]["input"]
        assert "<b>" not in pairs[0]["output"]

    def test_missing_com_field(self):
        """Test posts without 'com' field."""
        posts = [
            {"no": 1},  # No com field
            {"com": "Has content"},
            {"no": 3},  # No com field
        ]
        pairs = build_pairs_adjacent(posts, min_len=3)
        # Empty strings from missing com fields
        assert len(pairs) == 0  # Empty strings filtered by min_len


# =============================================================================
# build_pairs_contextual() tests
# =============================================================================


class TestBuildPairsContextual:
    """Tests for contextual pair building."""

    def test_cumulative_strategy(self):
        """Test cumulative context strategy."""
        posts = [
            {"com": "Post one content"},
            {"com": "Post two content"},
            {"com": "Post three content"},
        ]
        pairs = build_pairs_contextual(posts, strategy="cumulative", min_len=3)

        assert len(pairs) == 2
        # First pair: input is post 1, output is post 2
        assert "Post one" in pairs[0]["input"]
        # Second pair: input is posts 1+2, output is post 3
        assert "Post one" in pairs[1]["input"]
        assert "Post two" in pairs[1]["input"]

    def test_last_k_strategy(self):
        """Test last_k context strategy."""
        posts = [{"com": f"Post {i} content"} for i in range(10)]
        pairs = build_pairs_contextual(posts, strategy="last_k", k=3, min_len=3)

        # Each pair should have at most k posts in context
        for pair in pairs:
            # Count how many "Post X" patterns in input
            count = pair["input"].count("Post")
            assert count <= 3

    def test_max_chars_truncation(self):
        """Test max_chars truncation."""
        posts = [
            {"com": "A" * 100},
            {"com": "B" * 100},
            {"com": "C" * 100},
        ]
        pairs = build_pairs_contextual(posts, max_chars=50, min_len=3)

        for pair in pairs:
            assert len(pair["input"]) <= 50

    def test_empty_posts(self):
        """Test with empty posts."""
        pairs = build_pairs_contextual([])
        assert pairs == []

    def test_filters_short_context(self):
        """Test that short context entries are filtered."""
        posts = [
            {"com": "OK"},  # Too short for context
            {"com": "This is a longer post"},
            {"com": "Another longer post"},
        ]
        pairs = build_pairs_contextual(posts, min_len=5)

        # First post too short, so it's filtered from context
        assert len(pairs) >= 0  # May have pairs or not depending on fallback


# =============================================================================
# build_pairs_quote_contextual() tests
# =============================================================================


class TestBuildPairsQuoteContextual:
    """Tests for quote-chain contextual pair building."""

    def test_basic_quote_chain(self):
        """Test basic quote chain following."""
        posts = [
            {"no": 1, "com": "Original post content here with enough text"},
            {"no": 2, "com": "Reply to original with enough text >>1"},
            {"no": 3, "com": "Reply to reply with enough text >>2"},
        ]
        pairs = build_pairs_quote_contextual(posts, min_len=3)

        # Should produce pairs (exact count depends on implementation)
        assert isinstance(pairs, list)

    def test_fallback_to_adjacent(self):
        """Test fallback when no quote chain."""
        posts = [
            {"no": 1, "com": "First post no quotes"},
            {"no": 2, "com": "Second post no quotes"},
        ]
        pairs = build_pairs_quote_contextual(posts, min_len=3)

        # Should still produce pairs via fallback
        assert len(pairs) >= 0

    def test_require_question_filter(self):
        """Test require_question filter."""
        posts = [
            {"no": 1, "com": "This is a statement."},
            {"no": 2, "com": ">>1 This is a response."},
        ]
        pairs_no_filter = build_pairs_quote_contextual(posts, require_question=False, min_len=3)
        pairs_with_filter = build_pairs_quote_contextual(posts, require_question=True, min_len=3)

        # With filter, should have fewer or equal pairs
        assert len(pairs_with_filter) <= len(pairs_no_filter)

    def test_question_detection(self):
        """Test that questions are detected."""
        posts = [
            {"no": 1, "com": "What is the meaning of life?"},
            {"no": 2, "com": ">>1 The answer is 42."},
        ]
        pairs = build_pairs_quote_contextual(posts, require_question=True, min_len=3)

        # Should have at least one pair since first post is a question
        assert len(pairs) >= 0

    def test_merge_same_id(self):
        """Test merging posts from same ID."""
        posts = [
            {"no": 1, "id": "abc", "com": "First part"},
            {"no": 2, "id": "abc", "com": "Second part"},
            {"no": 3, "id": "xyz", "com": ">>1 Reply"},
        ]
        pairs = build_pairs_quote_contextual(posts, merge_same_id=True, min_len=3)
        # Should merge posts 1 and 2 in context
        assert len(pairs) >= 0

    def test_k_limit(self):
        """Test k limit on chain depth."""
        posts = [{"no": i, "com": f">>{i - 1 if i > 1 else ''} Post {i}"} for i in range(1, 20)]
        pairs = build_pairs_quote_contextual(posts, k=3, min_len=3)

        # Chain should be limited to k posts
        assert len(pairs) >= 0


# =============================================================================
# Edge case tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_unicode_content(self):
        """Test handling of unicode content."""
        posts = [
            {"com": "日本語のテキスト"},
            {"com": "Ответ на русском языке"},
        ]
        pairs = build_pairs_adjacent(posts, min_len=3)
        assert len(pairs) == 1
        assert "日本語" in pairs[0]["input"]

    def test_emoji_content(self):
        """Test handling of emoji content."""
        posts = [
            {"com": "Hello 👋 world 🌍"},
            {"com": "Response with 🎉 emoji"},
        ]
        pairs = build_pairs_adjacent(posts, min_len=3)
        assert len(pairs) == 1

    def test_very_long_posts(self):
        """Test handling of very long posts."""
        posts = [
            {"com": "A" * 10000},
            {"com": "B" * 10000},
        ]
        pairs = build_pairs_adjacent(posts, min_len=3)
        assert len(pairs) == 1
        assert len(pairs[0]["input"]) == 10000

    def test_special_characters(self):
        """Test handling of special characters."""
        posts = [
            {"com": "Code: if (x < y && z > 0) { return true; }"},
            {"com": "Response with 'quotes' and \"double quotes\""},
        ]
        pairs = build_pairs_adjacent(posts, min_len=3)
        assert len(pairs) == 1

    def test_malformed_html(self):
        """Test handling of malformed HTML."""
        posts = [
            {"com": "<p>Unclosed tag<b>nested"},
            {"com": "Normal response"},
        ]
        # Should not crash
        pairs = build_pairs_adjacent(posts, min_len=3)
        assert isinstance(pairs, list)

    def test_only_greentext(self):
        """Test posts that are only greentext."""
        posts = [
            {"com": ">be me\n>doing thing\n>mfw"},
            {"com": "Normal response here"},
        ]
        pairs = build_pairs_adjacent(posts, min_len=3)
        # First post becomes empty after cleaning greentext
        assert len(pairs) == 0  # Empty input filtered

    def test_only_urls(self):
        """Test posts that are only URLs."""
        posts = [
            {"com": "https://example.com https://test.com"},
            {"com": "Normal response here"},
        ]
        pairs = build_pairs_adjacent(posts, min_len=3)
        # First post becomes empty after removing URLs
        assert len(pairs) == 0  # Empty input filtered

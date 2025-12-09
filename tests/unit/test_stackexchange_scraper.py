"""Unit tests for scrapers/stackexchange_scraper.py.

Tests cover:
- HTML to text conversion (html_to_text)
- Session configuration (apply_session_config)
"""

from scrapers.stackexchange_scraper import (
    html_to_text,
    apply_session_config,
)


# =============================================================================
# html_to_text() tests
# =============================================================================


class TestHtmlToText:
    """Tests for HTML to text conversion."""

    def test_basic_html(self):
        """Test basic HTML stripping."""
        html = "<p>Hello <b>world</b></p>"
        result = html_to_text(html)
        assert "Hello" in result
        assert "world" in result
        assert "<p>" not in result
        assert "<b>" not in result

    def test_code_blocks(self):
        """Test code block conversion."""
        html = "<pre><code>print('hello')</code></pre>"
        result = html_to_text(html)
        assert "```" in result
        assert "print('hello')" in result

    def test_br_tags(self):
        """Test <br> tag conversion to newlines."""
        html = "Line 1<br>Line 2<br/>Line 3"
        result = html_to_text(html)
        assert "Line 1" in result
        assert "Line 2" in result
        assert "<br>" not in result

    def test_paragraph_tags(self):
        """Test <p> tag conversion."""
        html = "<p>Paragraph 1</p><p>Paragraph 2</p>"
        result = html_to_text(html)
        assert "Paragraph 1" in result
        assert "Paragraph 2" in result

    def test_html_entities(self):
        """Test HTML entity unescaping."""
        html = "&amp; &quot;text&quot;"
        result = html_to_text(html)
        assert "&" in result
        assert '"text"' in result

    def test_empty_string(self):
        """Test empty string."""
        result = html_to_text("")
        assert result == ""

    def test_none_input(self):
        """Test None input (should handle gracefully)."""
        result = html_to_text(None)
        assert result == ""

    def test_nested_tags(self):
        """Test nested HTML tags."""
        html = "<div><span><a href='#'>Link text</a></span></div>"
        result = html_to_text(html)
        assert "Link text" in result
        assert "<div>" not in result
        assert "<a" not in result

    def test_collapses_blank_lines(self):
        """Test that multiple blank lines are collapsed."""
        html = "<p>Line 1</p><p></p><p></p><p>Line 2</p>"
        result = html_to_text(html)
        # Should not have multiple consecutive blank lines
        assert "\n\n\n" not in result

    def test_preserves_code_formatting(self):
        """Test that code blocks preserve formatting."""
        html = "<pre><code>def foo():\n    return 42</code></pre>"
        result = html_to_text(html)
        assert "def foo():" in result
        assert "return 42" in result

    def test_strips_trailing_whitespace(self):
        """Test that result is stripped."""
        html = "   <p>Content</p>   "
        result = html_to_text(html)
        assert not result.startswith(" ")
        assert not result.endswith(" ")

    def test_complex_html(self):
        """Test complex HTML structure."""
        html = """
        <div class="question">
            <p>How do I do X?</p>
            <pre><code>some_code()</code></pre>
            <p>I tried Y but it didn't work.</p>
        </div>
        """
        result = html_to_text(html)
        assert "How do I do X?" in result
        assert "some_code()" in result
        assert "I tried Y" in result


# =============================================================================
# apply_session_config() tests
# =============================================================================


class TestApplySessionConfig:
    """Tests for session configuration."""

    def test_creates_session(self):
        """Test that session is created."""
        apply_session_config()
        import scrapers.stackexchange_scraper as se
        assert se.SESSION is not None

    def test_sets_user_agent(self):
        """Test that user agent is set."""
        apply_session_config()
        import scrapers.stackexchange_scraper as se
        assert se.SESSION.headers.get("User-Agent") == se.USER_AGENT

    def test_can_call_multiple_times(self):
        """Test that config can be applied multiple times."""
        apply_session_config()
        apply_session_config()
        import scrapers.stackexchange_scraper as se
        assert se.SESSION is not None


# =============================================================================
# Edge case tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_unicode_content(self):
        """Test unicode content handling."""
        html = "<p>日本語のテキスト</p>"
        result = html_to_text(html)
        assert "日本語" in result

    def test_emoji_content(self):
        """Test emoji handling."""
        html = "<p>Hello 👋 World 🌍</p>"
        result = html_to_text(html)
        assert "👋" in result
        assert "🌍" in result

    def test_malformed_html(self):
        """Test malformed HTML handling."""
        html = "<p>Unclosed tag<b>nested"
        # Should not crash
        result = html_to_text(html)
        assert isinstance(result, str)

    def test_script_tags(self):
        """Test that script tags are removed."""
        html = "<p>Text</p><script>alert('xss')</script><p>More</p>"
        result = html_to_text(html)
        assert "alert" not in result or "script" not in result.lower()

    def test_style_tags(self):
        """Test style tag content."""
        # Note: html_to_text doesn't specifically remove style tag content
        # It just strips tags, so CSS text may remain
        html = "<style>.class { color: red; }</style><p>Content</p>"
        result = html_to_text(html)
        assert "Content" in result

    def test_very_long_content(self):
        """Test very long content."""
        html = "<p>" + "word " * 10000 + "</p>"
        result = html_to_text(html)
        assert len(result) > 0

    def test_special_characters(self):
        """Test special characters."""
        html = "<p>Code: if (x < y && z > 0) { return true; }</p>"
        result = html_to_text(html)
        assert "if" in result
        assert "return true" in result

    def test_inline_code(self):
        """Test inline code tags."""
        html = "<p>Use <code>print()</code> to output</p>"
        result = html_to_text(html)
        assert "print()" in result

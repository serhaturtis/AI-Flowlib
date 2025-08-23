"""Tests for text formatting utilities."""

import pytest
from flowlib.utils.formatting.text import process_escape_sequences, truncate_text, format_key_value_pairs


class TestProcessEscapeSequences:
    """Test escape sequence processing."""
    
    def test_process_escape_sequences_basic(self):
        """Test basic escape sequence processing."""
        text = "Hello\\nWorld\\tTest"
        processed = process_escape_sequences(text)
        assert "\\n" in processed or "\n" in processed
        assert "\\t" in processed or "\t" in processed
    
    def test_process_escape_sequences_empty(self):
        """Test processing empty text."""
        assert process_escape_sequences("") == ""
    
    def test_process_escape_sequences_no_escapes(self):
        """Test text with no escape sequences."""
        text = "Hello World"
        processed = process_escape_sequences(text)
        assert processed == text


class TestTruncateText:
    """Test text truncation utilities."""
    
    def test_truncate_text_basic(self):
        """Test basic text truncation."""
        long_text = "This is a very long text that should be truncated"
        
        # Basic truncation
        truncated = truncate_text(long_text, max_length=20)
        assert len(truncated) <= 23  # Including ellipsis
        assert truncated.endswith("...")
    
    def test_truncate_text_no_ellipsis(self):
        """Test truncation without ellipsis."""
        long_text = "This is a very long text that should be truncated"
        truncated = truncate_text(long_text, max_length=20, add_ellipsis=False)
        assert len(truncated) == 20
        assert not truncated.endswith("...")
    
    def test_truncate_text_short(self):
        """Test text shorter than limit."""
        short_text = "Short"
        unchanged = truncate_text(short_text, max_length=20)
        assert unchanged == "Short"
    
    def test_truncate_text_edge_cases(self):
        """Test text truncation edge cases."""
        # Empty text
        assert truncate_text("", 10) == ""


class TestFormatKeyValuePairs:
    """Test key-value pair formatting."""
    
    def test_format_key_value_pairs_basic(self):
        """Test basic key-value formatting."""
        data = {"name": "test", "value": 42, "enabled": True}
        formatted = format_key_value_pairs(data)
        assert isinstance(formatted, str)
        assert "name" in formatted
        assert "test" in formatted
    
    def test_format_key_value_pairs_empty(self):
        """Test formatting empty dictionary."""
        formatted = format_key_value_pairs({})
        assert isinstance(formatted, str)
    
    def test_format_key_value_pairs_nested(self):
        """Test formatting nested data."""
        data = {"outer": {"inner": "value"}, "list": [1, 2, 3]}
        formatted = format_key_value_pairs(data)
        assert isinstance(formatted, str)
        assert "outer" in formatted
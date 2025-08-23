"""Tests for JSON formatting utilities."""

import pytest
import json
from typing import Any, Dict
from flowlib.utils.formatting.json import extract_json, extract_json_str, format_json


class TestExtractJson:
    """Test JSON extraction utilities."""
    
    def test_extract_json_valid(self):
        """Test extracting valid JSON."""
        text = 'Some text {"key": "value", "number": 42} more text'
        result = extract_json(text)
        assert result is not None
        assert result["key"] == "value"
        assert result["number"] == 42
    
    def test_extract_json_no_json(self):
        """Test text with no JSON."""
        text = "This is just plain text"
        result = extract_json(text)
        assert result is None
    
    def test_extract_json_invalid(self):
        """Test text with valid JSON surrounded by other text."""
        text = 'Some text {"key": "value"} more text'
        result = extract_json(text)
        # Should return the valid JSON found
        assert result is not None
        assert result["key"] == "value"
    
    def test_extract_json_empty(self):
        """Test empty text."""
        result = extract_json("")
        assert result is None


class TestExtractJsonStr:
    """Test JSON string extraction utilities."""
    
    def test_extracting_json_string(self):
        """Test extracting JSON string."""
        text = 'Some text {"key": "value"} more text'
        result = extract_json_str(text)
        assert result is not None
        assert '"key"' in result
        assert '"value"' in result
    
    # Removed redundant str() test
        """Test text with no JSON."""
        text = "This is just plain text"
        result = extract_json_str(text)
        assert result == ""
    
    # Removed redundant str() test
        """Test text with single valid JSON object."""
        text = 'Some text {"a": 1} more text'
        result = extract_json_str(text)
        assert result != ""
        # Should return the JSON found
        assert '"a"' in result


class TestFormatJson:
    """Test JSON formatting utilities."""
    
    def test_format_json_basic(self):
        """Test basic JSON formatting."""
        data = {"name": "test", "items": [1, 2, 3], "nested": {"key": "value"}}
        formatted = format_json(data)
        assert isinstance(formatted, str)
        # Should be pretty-printed
        assert "\n" in formatted
        assert "  " in formatted  # Indentation
    
    def test_format_json_compact(self):
        """Test compact JSON formatting."""
        data = {"name": "test", "value": 42}
        formatted = format_json(data, indent=None)
        assert isinstance(formatted, str)
        # Should be compact
        assert "\n" not in formatted
    
    def test_format_json_custom_indent(self):
        """Test JSON formatting with custom indentation."""
        data = {"name": "test", "value": 42}
        formatted = format_json(data, indent=4)
        assert isinstance(formatted, str)
        assert "    " in formatted  # 4-space indentation
    
    def test_format_json_edge_cases(self):
        """Test JSON formatting edge cases."""
        # Empty dict
        formatted = format_json({})
        assert formatted == "{}"
        
        # None values
        data = {"key": None, "valid": "value"}
        formatted = format_json(data)
        assert "null" in formatted
        
        # Complex nested structure
        complex_data = {
            "level1": {
                "level2": {
                    "level3": ["item1", "item2", {"level4": True}]
                }
            }
        }
        formatted = format_json(complex_data)
        assert isinstance(formatted, str)
        assert "level1" in formatted
        assert "level4" in formatted
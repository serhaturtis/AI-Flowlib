"""JSON formatting utilities.

This module provides utilities for extracting and formatting JSON data
from text responses, particularly useful for handling LLM outputs.
"""

import json
import re
from typing import Any


def extract_json(text: str) -> dict[str, Any] | list[Any] | None:
    """Extract JSON data from text.

    This function attempts to find and parse a JSON object or array
    within the provided text. It handles both complete JSON responses
    and JSON embedded within other text.

    Args:
        text: Text that may contain JSON

    Returns:
        Extracted JSON data as a Python dict/list or None if extraction fails
    """
    if not text:
        return None

    # Try to find a JSON object using regex
    json_pattern = r"(\{[\s\S]*\}|\[[\s\S]*\])"
    match = re.search(json_pattern, text)

    if match:
        json_str = match.group(0)
        try:
            result = json.loads(json_str)
            # Only return dict or list, filter out other JSON types
            if isinstance(result, (dict, list)):
                return result
        except json.JSONDecodeError:
            pass

    # If no valid JSON found with regex, try the whole text
    try:
        result = json.loads(text)
        # Only return dict or list, filter out other JSON types
        if isinstance(result, (dict, list)):
            return result
    except json.JSONDecodeError:
        pass

    return None


def extract_json_str(text: str) -> str:
    """Extract JSON string from text.

    Similar to extract_json but returns the raw JSON string instead
    of parsing it into Python objects.

    Args:
        text: Text that may contain JSON

    Returns:
        Extracted JSON string or empty string if none found
    """
    if not text:
        return ""

    # Try to find a JSON object using regex
    json_pattern = r"(\{[\s\S]*\}|\[[\s\S]*\])"
    match = re.search(json_pattern, text)

    if match:
        json_str = match.group(0)
        try:
            # Validate it's actually JSON
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            pass

    # If no valid JSON found with regex, try the whole text
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    return ""


def format_json(data: dict[str, Any] | list[Any], indent: int = 2) -> str:
    """Format Python data as pretty-printed JSON.

    Args:
        data: Data to format as JSON
        indent: Number of spaces for indentation

    Returns:
        Formatted JSON string
    """
    if not data:
        return "{}" if isinstance(data, dict) else "[]"

    try:
        return json.dumps(data, indent=indent, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        # Handle non-serializable objects
        return f"Error formatting as JSON: {str(e)}"

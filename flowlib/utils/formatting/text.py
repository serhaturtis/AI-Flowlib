"""Text formatting utilities.

This module provides utilities for processing and formatting text data,
including handling escape sequences and other common text transformations.
"""

import re
from typing import List, Optional, Dict, Any


def process_escape_sequences(text: str) -> str:
    """Process any literal escape sequences in the text.
    
    This method replaces literal escape sequences like '\\n' with
    their actual character representation.
    
    Args:
        text: Text to process
            
    Returns:
        Processed text with proper escape sequences
    """
    if not text:
        return ""
        
    # Replace common literal escape sequences with their character representation
    # Start with double backslashes (\\n) as they may appear in JSON strings
    text = text.replace('\\\\n', '\n')
    text = text.replace('\\n', '\n')
    text = text.replace('\\t', '\t')
    text = text.replace('\\r', '\r')
    
    # Handle unicode escape sequences like \u00A0
    text = re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), text)
    
    return text


def truncate_text(text: str, max_length: int, add_ellipsis: bool = True) -> str:
    """Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length in characters
        add_ellipsis: Whether to add ellipsis (...) when truncated
            
    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text
        
    truncated = text[:max_length]
    if add_ellipsis:
        # Replace last three chars with ellipsis if needed
        truncated = truncated[:-3] + "..."
        
    return truncated


def format_key_value_pairs(pairs: Dict[str, Any], delimiter: str = ": ") -> str:
    """Format key-value pairs into a string.
    
    Args:
        pairs: Dictionary of key-value pairs
        delimiter: Delimiter between key and value
            
    Returns:
        Formatted string with one pair per line
    """
    if not pairs:
        return ""
        
    lines = []
    for key, value in pairs.items():
        lines.append(f"{key}{delimiter}{value}")
        
    return "\n".join(lines) 
"""Formatting utilities for flowlib.

This package provides utilities for formatting various types of data in the flowlib ecosystem,
including text processing, entity formatting, conversation formatting, and JSON operations.
"""

# Import all formatters for easy access
from flowlib.utils.formatting.text import process_escape_sequences, format_key_value_pairs
from flowlib.utils.formatting.entities import (
    format_entity_for_display,
    format_entities_as_context,
    format_entity_list
)
from flowlib.utils.formatting.conversation import (
    format_conversation,
    format_state,
    format_history,
    format_flows,
    format_agent_execution_details,
    format_execution_history
)
from flowlib.utils.formatting.json import extract_json, format_json
from flowlib.utils.pydantic.schema import get_model_schema_text, model_to_simple_json_schema
from flowlib.utils.formatting.serialization import make_serializable, format_execution_details

__all__ = [
    # Text formatting
    "process_escape_sequences",
    "format_key_value_pairs",
    
    # Entity formatting
    "format_entity_for_display",
    "format_entities_as_context",
    "format_entity_list",
    
    # Conversation formatting
    "format_conversation",
    "format_state",
    "format_history",
    "format_flows",
    "format_agent_execution_details",
    "format_execution_history",
    
    # JSON formatting
    "extract_json",
    "format_json",
    
    # Schema formatting
    "get_model_schema_text",
    "model_to_simple_json_schema",
    
    # Serialization
    "make_serializable",
    "format_execution_details"
] 
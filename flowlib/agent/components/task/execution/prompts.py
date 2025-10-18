"""Tool selection prompts for agent tool calling.

This module provides prompt templates for LLM-driven tool selection
following flowlib's prompt resource pattern.
"""

from typing import Any, Dict

from flowlib.resources.decorators.decorators import prompt
from flowlib.resources.models.base import ResourceBase


@prompt("intent-classification")
class IntentClassificationPrompt(ResourceBase):
    """Prompt for classifying user intent before tool selection."""

    template: str = """Message: {{user_message}}

ACTION: User wants me to DO something (execute task, perform operation)
CONVERSATION: User wants me to SAY something (provide information, discuss, explain)

Classify as ACTION or CONVERSATION."""

    config: Dict[str, Any] = {
        "max_tokens": 200,
        "temperature": 0.1,
        "top_p": 0.9
    }


@prompt("tool-selection")
class ToolSelectionPrompt(ResourceBase):
    """Prompt for LLM tool selection with structured output."""

    template: str = """Request: {{task_description}}

Tools:
{{available_tools}}

Select tools for operations. Use no tools for conversation."""

    config: Dict[str, Any] = {
        "max_tokens": 2048,
        "temperature": 0.3,
        "top_p": 0.9
    }
